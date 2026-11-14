"""CHP2 session / CHPB packets, network-endian uint16 IDs, CRC32, explicit EOS.

All bytes count, including the bootstrap, session, and every incremental header.
The v2 transport is ordered within and across blocks. Each final block prefix
explicitly closes the block, without looking at a future packet.
"""

import json
import struct
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .config import CODEBOOK_SIZE, FSQ_LEVELS
from .model import Geometry

SESSION = struct.Struct(">4sI")
PACKET = struct.Struct(">4sIQHHHIB")
CRC = struct.Struct(">I")
MAX_HEADER = 16384


@dataclass(frozen=True)
class Session:
    model_id: str
    checkpoint_sha256: str
    config_sha256: str
    width: int
    height: int
    padded_width: int
    padded_height: int
    kmax: int = 2048
    fps_num: int = 30
    fps_den: int = 1
    protocol_version: int = 2
    temporal_block_size: int = 4
    bootstrap_mode: str = "one-discrete-frame"
    quantizer_type: str = "FSQ"
    fsq_levels: tuple[int, ...] = FSQ_LEVELS
    bits_per_token: int = 16
    timebase_num: int = 1
    timebase_den: int = 30

    def __post_init__(self):
        object.__setattr__(self, "fsq_levels", tuple(self.fsq_levels))
        Geometry(self.height, self.width, self.padded_height, self.padded_width)
        self._validate_protocol()
        self._validate_timing()
        self._validate_identity()

    def _validate_protocol(self):
        settings = (
            self.protocol_version,
            self.temporal_block_size,
            self.bits_per_token,
            self.fsq_levels,
            self.quantizer_type,
            self.bootstrap_mode,
        )
        if settings != (2, 4, 16, FSQ_LEVELS, "FSQ", "one-discrete-frame"):
            raise ValueError("unsupported CHP session")
        if self.kmax not in (2048, 4096, 8192):
            raise ValueError("unsupported Kmax")

    def _validate_timing(self):
        if min(self.fps_num, self.fps_den, self.timebase_num, self.timebase_den) <= 0:
            raise ValueError("invalid timing")
        if self.timebase_num * self.fps_num != self.timebase_den * self.fps_den:
            raise ValueError("v2 timebase must be inverse fps")

    def _validate_identity(self):
        for sha in (self.checkpoint_sha256, self.config_sha256):
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid SHA256")
        if not self.model_id or len(self.model_id) > 128:
            raise ValueError("invalid model ID")

    @property
    def geometry(self):
        return Geometry(self.height, self.width, self.padded_height, self.padded_width)

    def to_bytes(self):
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        header = SESSION.pack(b"CHP2", len(payload)) + payload
        return header + CRC.pack(zlib.crc32(header))

    @classmethod
    def read(cls, source):
        header = exact(source, SESSION.size)
        magic, size = SESSION.unpack(header)
        if magic != b"CHP2" or size > MAX_HEADER:
            raise ValueError("invalid CHP2 header")
        payload = exact(source, size)
        if CRC.unpack(exact(source, CRC.size))[0] != zlib.crc32(header + payload):
            raise ValueError("session CRC32 mismatch")
        try:
            return cls(**json.loads(payload))
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError("invalid session metadata") from e


def exact(source, count):
    data = source.read(count)
    if len(data) != count:
        raise ValueError("truncated CHP stream")
    return data


@dataclass(frozen=True)
class Packet:
    block_id: int
    frame_start: int
    frame_count: int
    prefix_offset: int
    ids: tuple[int, ...]
    eos: bool = False
    block_end: bool = False
    tail: bool = False

    def __post_init__(self):
        if self.eos:
            object.__setattr__(self, "block_end", True)
        self._validate_frame_flags()
        self._validate_counts()
        if any(
            not isinstance(i, (int, np.integer)) or not 0 <= i < CODEBOOK_SIZE
            for i in self.ids
        ):
            raise ValueError("invalid FSQ symbol")

    def _validate_frame_flags(self):
        if self.tail and (self.block_id == 0 or self.frame_count == 4):
            raise ValueError("tail flag requires a partial final temporal block")
        if not 0 <= self.block_id < 2**32 or not 0 <= self.frame_start < 2**64:
            raise ValueError("invalid timestamp/block")

    def _validate_counts(self):
        if not 1 <= self.frame_count <= 4:
            raise ValueError("invalid frame count")
        if not 0 <= self.prefix_offset < 8192 or not 1 <= len(self.ids) <= 8192:
            raise ValueError("invalid packet counts")

    def to_bytes(self):
        payload = np.asarray(self.ids, dtype=">u2").tobytes()
        header = PACKET.pack(
            b"CHPB",
            self.block_id,
            self.frame_start,
            self.frame_count,
            self.prefix_offset,
            len(self.ids),
            len(payload),
            int(self.eos) | (int(self.block_end) << 1) | (int(self.tail) << 2),
        )
        return header + payload + CRC.pack(zlib.crc32(header + payload))

    @classmethod
    def read(cls, source):
        header = source.read(PACKET.size)
        if not header:
            return None
        if len(header) != PACKET.size:
            raise ValueError("truncated packet header")
        magic, block, start, count, offset, size, length, flags = PACKET.unpack(header)
        if (
            magic != b"CHPB"
            or size > 8192
            or length != size * 2
            or flags not in range(8)
        ):
            raise ValueError("invalid packet header")
        payload = exact(source, length)
        if CRC.unpack(exact(source, CRC.size))[0] != zlib.crc32(header + payload):
            raise ValueError("packet CRC32 mismatch")
        return cls(
            block,
            start,
            count,
            offset,
            tuple(np.frombuffer(payload, dtype=">u2").tolist()),
            bool(flags & 1),
            bool(flags & 2),
            bool(flags & 4),
        )


class PrefixReceiver:
    def __init__(self, session, *, model_id, checkpoint_sha256, config_sha256):
        if (session.model_id, session.checkpoint_sha256, session.config_sha256) != (
            model_id,
            checkpoint_sha256,
            config_sha256,
        ):
            raise ValueError("model/checkpoint/config identity mismatch")
        self.session = session
        self.blocks, self.metadata = [], []
        self.actual_bytes = len(session.to_bytes())
        self.finished = False
        self.block_closed = False

    def _validate_position(self, packet):
        if self.finished:
            raise ValueError("tokens after EOS")
        if packet.block_id > len(self.blocks) or packet.block_id < len(self.blocks) - 1:
            raise ValueError("out-of-order or late block")
        self._validate_prefix_count(packet)
        self._validate_frame_metadata(packet)

    def _validate_prefix_count(self, packet):
        end = packet.prefix_offset + len(packet.ids)
        if end > self.session.kmax:
            raise ValueError("prefix exceeds Kmax")
        if packet.block_end and end < 256:
            raise ValueError("block end/EOS requires a base prefix")
        if packet.tail and packet.block_end and not packet.eos:
            raise ValueError("a partial tail must finish with explicit EOS")

    def _validate_new_block(self, packet):
        if self.blocks and len(self.blocks[-1]) < 256:
            raise ValueError("cannot advance without base prefix")
        if self.blocks and not self.block_closed:
            raise ValueError("previous block must be explicitly closed")
        if packet.prefix_offset != 0:
            raise ValueError("missing necessary prefix")

    @staticmethod
    def _validate_frame_metadata(packet):
        expected = 0 if packet.block_id == 0 else 1 + 4 * (packet.block_id - 1)
        if packet.frame_start != expected:
            raise ValueError("frame alignment error")
        if packet.block_id == 0 and packet.frame_count != 1:
            raise ValueError("bootstrap frame count error")
        if (
            packet.block_id > 0
            and packet.frame_count != 4
            and not (packet.eos or packet.tail)
        ):
            raise ValueError("partial tail needs EOS")

    def _validate_existing_block(self, packet):
        if self.block_closed:
            raise ValueError("block already closed")
        if self.metadata[packet.block_id] != (packet.frame_start, packet.frame_count):
            raise ValueError("block geometry changed")
        if packet.prefix_offset != len(self.blocks[packet.block_id]):
            raise ValueError("duplicate or non-contiguous prefix offset")

    def append_tokens(self, packet):
        self._validate_position(packet)
        new_block = packet.block_id == len(self.blocks)
        if new_block:
            self._validate_new_block(packet)
        else:
            self._validate_existing_block(packet)
        # All validation precedes mutation, including rejected first packets.
        if new_block:
            self.blocks.append([])
            self.metadata.append((packet.frame_start, packet.frame_count))
        self.blocks[packet.block_id].extend(packet.ids)
        self.actual_bytes += len(packet.to_bytes())
        self.finished = packet.eos
        self.block_closed = packet.block_end
        return torch.tensor([self.blocks[packet.block_id]], dtype=torch.int32)

    def read_packet(self, source):
        packet = Packet.read(source)
        if packet is None:
            if not self.finished:
                raise ValueError("missing explicit EOS")
            return None
        self.append_tokens(packet)
        return packet
