"""Use the same framed bytes for file and ordered TCP transport."""

import socket
import threading
import pytest
from cosmos_predict1.tokenizer.progressive.transport import open_transport
from cosmos_predict1.tokenizer.progressive.protocol import Session, Packet


def test_socket_transports_exact_framed_prefix_bytes():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    endpoint = f"tcp://127.0.0.1:{port}"
    data = (
        Session("test", "a" * 64, "b" * 64, 64, 32, 64, 32).to_bytes()
        + Packet(0, 0, 1, 0, (0, 63999) * 128, True).to_bytes()
    )
    received = []

    def receive():
        with open_transport(endpoint, "rb") as source:
            session = Session.read(source)
            packet = Packet.read(source)
            assert Packet.read(source) is None
            received.extend([session.to_bytes(), packet.to_bytes()])

    thread = threading.Thread(target=receive)
    thread.start()
    with open_transport(endpoint, "xb") as target:
        target.write(data)
        target.flush()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert b"".join(received) == data


def test_file_transport_refuses_overwrite(tmp_path):
    target = tmp_path / "stream.chp"
    with open_transport(str(target), "xb") as f:
        f.write(b"abc")
    with open_transport(str(target), "rb") as f:
        assert f.read() == b"abc"
    with pytest.raises(FileExistsError):
        with open_transport(str(target), "xb"):
            pass


def test_sender_waits_for_delayed_receiver_start():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    received = []

    def delayed_receiver():
        threading.Event().wait(0.06)
        with open_transport(endpoint, "rb", timeout=2) as source:
            received.append(source.read(5))

    thread = threading.Thread(target=delayed_receiver)
    thread.start()
    with open_transport(endpoint, "xb", timeout=2) as target:
        target.write(b"CHP2!")
    thread.join(timeout=3)
    assert not thread.is_alive() and received == [b"CHP2!"]


def test_sender_connection_refusal_has_a_finite_deadline():
    # A bound, non-listening port cannot be taken by another local process.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
        with pytest.raises(ConnectionRefusedError):
            with open_transport(endpoint, "xb", timeout=0.02):
                pass
