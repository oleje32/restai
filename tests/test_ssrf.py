"""SSRF classifier tests for restai.helper._is_private_ip.

No network access: socket.getaddrinfo is monkeypatched to return controlled
addrinfo tuples so the IP classification can be exercised deterministically.
"""
import socket

import pytest

from restai.helper import _is_private_ip


def _addrinfo_for(*addresses):
    """Build getaddrinfo-shaped tuples for the given IP strings.

    Each entry mirrors socket.getaddrinfo's 5-tuple; only entry[4][0] (the
    address) is consulted by _is_private_ip.
    """
    out = []
    for addr in addresses:
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        sockaddr = (addr, 0, 0, 0) if family == socket.AF_INET6 else (addr, 0)
        out.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
    return out


@pytest.fixture
def patched_resolve(monkeypatch):
    def _install(*addresses):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *a, **k: _addrinfo_for(*addresses)
        )

    return _install


def test_loopback_is_private(patched_resolve):
    patched_resolve("127.0.0.1")
    assert _is_private_ip("evil.example.com") is True


def test_ipv4_mapped_ipv6_loopback_is_private(patched_resolve):
    patched_resolve("::ffff:127.0.0.1")
    assert _is_private_ip("evil.example.com") is True


def test_unspecified_zero_is_private(patched_resolve):
    patched_resolve("0.0.0.0")
    assert _is_private_ip("evil.example.com") is True


def test_cgnat_100_64_is_private(patched_resolve):
    patched_resolve("100.64.0.1")
    assert _is_private_ip("evil.example.com") is True


def test_ipv6_link_local_is_private(patched_resolve):
    patched_resolve("fe80::1")
    assert _is_private_ip("evil.example.com") is True


def test_public_address_is_not_private(patched_resolve):
    patched_resolve("93.184.216.34")  # example.com
    assert _is_private_ip("example.com") is False


def test_unresolvable_raises(monkeypatch):
    def _boom(*a, **k):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    with pytest.raises(ValueError):
        _is_private_ip("does-not-resolve.invalid")
