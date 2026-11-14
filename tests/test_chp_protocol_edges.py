import io
import struct
import zlib

import pytest

from cosmos_predict1.tokenizer.progressive.protocol import (
    PACKET,
    Packet,
    PrefixReceiver,
    Session,
)


def fresh():
    s = Session("m", "a" * 64, "b" * 64, 1920, 1080, 1920, 1088)
    return PrefixReceiver(
        s, model_id="m", checkpoint_sha256="a" * 64, config_sha256="b" * 64
    )


def test_rejects_late_skipped_unfinished_and_malformed_packets():
    r = fresh()
    with pytest.raises(ValueError, match="order"):
        r.append_tokens(Packet(1, 1, 4, 0, (0,) * 256))
    r.append_tokens(Packet(0, 0, 1, 0, (0,) * 128))
    with pytest.raises(ValueError, match="base"):
        r.append_tokens(Packet(1, 1, 4, 0, (0,) * 256))
    r.append_tokens(Packet(0, 0, 1, 128, (0,) * 128))
    r.append_tokens(Packet(1, 1, 4, 0, (0,) * 256))
    with pytest.raises(ValueError, match="late"):
        r.append_tokens(Packet(0, 0, 1, 256, (0,) * 256))
    with pytest.raises(ValueError, match="EOS"):
        r.read_packet(io.BytesIO())
    for data in (b"C", b"CHPB", Packet(0, 0, 1, 0, (0,) * 256).to_bytes()[:-1]):
        with pytest.raises(ValueError):
            Packet.read(io.BytesIO(data))
    for field in (
        dict(bits_per_token=13),
        dict(fsq_levels=(8,) * 6),
        dict(padded_height=1080),
        dict(checkpoint_sha256="x" * 64),
    ):
        args = dict(
            model_id="m",
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            width=1920,
            height=1080,
            padded_width=1920,
            padded_height=1088,
        )
        args.update(field)
        with pytest.raises(ValueError):
            Session(**args)


def test_explicit_eos_and_actual_tail_count():
    r = fresh()
    r.append_tokens(Packet(0, 0, 1, 0, (0,) * 256))
    with pytest.raises(ValueError, match="EOS"):
        r.append_tokens(Packet(1, 1, 3, 0, (0,) * 256))
    r.append_tokens(Packet(1, 1, 3, 0, (0,) * 256, True))
    assert r.finished and r.metadata[-1] == (1, 3)
    assert r.read_packet(io.BytesIO()) is None


def test_wire_out_of_range_ids_even_with_valid_crc_are_rejected():
    p = Packet(0, 0, 1, 0, (0,)).to_bytes()
    body = p[: PACKET.size] + struct.pack(">H", 64000)
    with pytest.raises(ValueError, match="FSQ"):
        Packet.read(io.BytesIO(body + struct.pack(">I", zlib.crc32(body))))


def test_rejected_packet_does_not_mutate_receiver_state():
    r = fresh()
    r.append_tokens(Packet(0, 0, 1, 0, (0,) * 256))
    before = (r.actual_bytes, list(map(list, r.blocks)), list(r.metadata))
    with pytest.raises(ValueError):
        r.append_tokens(Packet(1, 1, 4, 0, (0,) * 4096))
    assert (r.actual_bytes, r.blocks, r.metadata) == before
