"""A block boundary must be explicit, without reading a future packet."""

import io
import pytest
from cosmos_predict1.tokenizer.progressive.protocol import (
    Packet,
    Session,
    PrefixReceiver,
)


def test_final_prefix_is_self_describing_without_lookahead():
    packet = Packet(0, 0, 1, 0, tuple([0] * 256), block_end=True)
    restored = Packet.read(io.BytesIO(packet.to_bytes()))
    assert restored.block_end and not restored.eos
    s = Session("m", "a" * 64, "b" * 64, 64, 32, 64, 32)
    r = PrefixReceiver(
        s, model_id="m", checkpoint_sha256="a" * 64, config_sha256="b" * 64
    )
    r.append_tokens(restored)
    with pytest.raises(ValueError, match="closed"):
        r.append_tokens(Packet(0, 0, 1, 256, (0,) * 256))
    r.append_tokens(Packet(1, 1, 4, 0, (0,) * 256, block_end=True, eos=True))
    assert r.finished


def test_next_block_cannot_finalize_an_unclosed_previous_block():
    s = Session("m", "a" * 64, "b" * 64, 64, 32, 64, 32)
    r = PrefixReceiver(
        s, model_id="m", checkpoint_sha256="a" * 64, config_sha256="b" * 64
    )
    r.append_tokens(Packet(0, 0, 1, 0, (0,) * 256))
    with pytest.raises(ValueError, match="closed"):
        r.append_tokens(Packet(1, 1, 4, 0, (0,) * 256, block_end=True))
