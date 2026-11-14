"""Ordered byte transport; file IPC and TCP use the identical framed protocol."""

from contextlib import contextmanager
import socket
import time
from urllib.parse import urlparse


@contextmanager
def open_transport(endpoint, mode, timeout=30.0):
    if not endpoint.startswith("tcp://"):
        with open(endpoint, mode) as stream:
            yield stream
        return
    address = urlparse(endpoint)
    if address.hostname is None or address.port is None or address.path:
        raise ValueError("expected tcp://host:port")
    if mode == "rb":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(timeout)
            listener.bind((address.hostname, address.port))
            listener.listen(1)
            connection, _ = listener.accept()
    elif mode == "xb":
        connection = _connect(address.hostname, address.port, timeout)
    else:
        raise ValueError("transport is read-only rb or new stream xb")
    with connection:
        connection.settimeout(timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with connection.makefile("rb" if mode == "rb" else "wb") as stream:
            yield stream


def _connect(host, port, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=timeout)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
