"""Stateless per-chat agentic-browser runtime.

Docker daemon is the source of truth for "does this chat have a browser
container" — looked up by the ``restai.browser_chat_id`` label per call.
Prior in-memory state drifted between uvicorn workers and made the
settings-reinit path racy.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile
import threading
import time
from typing import Optional

import docker as _docker_sdk
import requests

from restai import config as _cfg

logger = logging.getLogger(__name__)

_CONTAINER_PORT = 7000
_MICRO_SERVER_PATH_HOST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_server.py")
_MICRO_SERVER_PATH_CONTAINER = "/opt/restai_browser/micro_server.py"
_HEALTH_TIMEOUT = 60
_STORAGE_STATE_REDIS_PREFIX = "restai_browser_state:"
_STORAGE_STATE_TTL = 30 * 24 * 60 * 60

_DEFAULT_IMAGE = "mcr.microsoft.com/playwright/python:v1.48.0-jammy"

_client: Optional[_docker_sdk.DockerClient] = None
_client_url: str = ""
_client_lock = threading.Lock()

_storage_local: dict[str, dict] = {}
_storage_redis_client = None
_storage_redis_url: Optional[str] = None

# Per-chat asyncio lock so two parallel browser_* calls in the same chat
# don't race-create two containers.
_chat_locks: dict[str, asyncio.Lock] = {}

# Sentinel project id for callers with no real project. Two DIFFERENT real
# projects never collide because their ids differ; only genuinely
# project-less callers share this bucket.
EPHEMERAL_PROJECT_ID = 0


def _scoped_id(project_id, chat_id: str) -> str:
    """Container identity key: namespaces the client-supplied chat_id by
    project so a chat_id from project A can never resolve to project B's
    browser container. Used verbatim as the `restai.browser_chat_id` label
    value AND the `browser_chat_activity` key, so create/resolve/cleanup all
    agree."""
    pid = EPHEMERAL_PROJECT_ID if project_id is None else int(project_id)
    return f"{pid}:{chat_id or 'ephemeral'}"


def is_enabled() -> bool:
    if not bool(getattr(_cfg, "BROWSER_ENABLED", False)):
        return False
    return bool((getattr(_cfg, "DOCKER_URL", "") or "").strip())


def _get_client() -> Optional[_docker_sdk.DockerClient]:
    """Cached DockerClient; rebuilt on docker_url change."""
    global _client, _client_url
    if not is_enabled():
        return None
    url = (getattr(_cfg, "DOCKER_URL", "") or "").strip()
    with _client_lock:
        if _client is not None and _client_url == url:
            return _client
        try:
            _client = _docker_sdk.DockerClient(base_url=url)
            _client_url = url
            logger.info("Browser Docker client connected to %s", url)
        except Exception as e:
            logger.error("Browser Docker client failed to connect to %s: %s", url, e)
            _client = None
            _client_url = ""
        return _client


def _redis():
    global _storage_redis_client, _storage_redis_url
    url = _cfg.build_redis_url()
    if not url:
        if _storage_redis_client is not None:
            try:
                _storage_redis_client.close()
            except Exception:
                pass
            _storage_redis_client = None
            _storage_redis_url = None
        return None
    if _storage_redis_client is not None and _storage_redis_url == url:
        return _storage_redis_client
    try:
        import redis
        _storage_redis_client = redis.Redis.from_url(url)
        _storage_redis_url = url
    except Exception as e:
        logger.warning("Browser: Redis unavailable (%s); using in-process storage-state fallback", e)
        return None
    return _storage_redis_client


def load_storage_state(project_id: int, domain: str) -> Optional[dict]:
    key = f"{_STORAGE_STATE_REDIS_PREFIX}{project_id}:{domain}"
    r = _redis()
    if r is not None:
        try:
            raw = r.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("Browser: failed to load storage_state from Redis (%s)", e)
    return _storage_local.get(key)


def save_storage_state(project_id: int, domain: str, state: dict) -> None:
    key = f"{_STORAGE_STATE_REDIS_PREFIX}{project_id}:{domain}"
    r = _redis()
    if r is not None:
        try:
            r.set(key, json.dumps(state), ex=_STORAGE_STATE_TTL)
            return
        except Exception as e:
            logger.warning("Browser: failed to save storage_state to Redis (%s)", e)
    _storage_local[key] = state


def _resolve_container(scoped_id: str):
    c = _get_client()
    if c is None or not scoped_id:
        return None
    try:
        matches = c.containers.list(
            filters={"label": [f"restai.browser_chat_id={scoped_id}", "restai.browser_managed=true"]},
            limit=1,
        )
    except Exception as e:
        logger.warning("Browser container list failed for chat_id=%s: %s", scoped_id, e)
        return None
    if matches and matches[0].status == "running":
        return matches[0]
    return None


def _discover_port(container) -> Optional[int]:
    try:
        binding = container.attrs["NetworkSettings"]["Ports"].get(f"{_CONTAINER_PORT}/tcp")
        if not binding:
            return None
        for b in binding:
            if b.get("HostIp") in ("127.0.0.1", "0.0.0.0", ""):
                return int(b["HostPort"])
    except Exception:
        return None
    return None


def _parse_playwright_version(image: str) -> Optional[str]:
    """Pin in-container pip install to the version whose browser binaries are baked in."""
    try:
        tag = image.split(":", 1)[1]
    except (IndexError, AttributeError):
        return None
    if tag.startswith("v"):
        tag = tag[1:]
    version = tag.split("-", 1)[0].split("+", 1)[0]
    parts = version.split(".")
    if len(parts) >= 2 and all(p.isdigit() for p in parts):
        return version
    return None


def _ensure_playwright_pkg(container, image: str) -> None:
    """Install the `playwright` pip package — the MS image ships binaries but not the pkg."""
    probe = container.exec_run(["sh", "-c", "python3 -c 'import playwright' 2>/dev/null"])
    if probe.exit_code == 0:
        return
    pin = _parse_playwright_version(image)
    if not pin:
        # Fail closed rather than `pip install playwright` (unpinned latest):
        # an unpinned runtime install is a supply-chain foothold and may also
        # mismatch the browser binaries baked into the image. Require a
        # version-tagged image, or bake the `playwright` pkg into a custom image
        # (recommended — then set BROWSER_READ_ONLY=true to drop runtime pip).
        raise RuntimeError(
            f"Cannot pin playwright version from image tag '{image}'. Use a "
            f"version-tagged Playwright image (e.g. ...:v1.48.0-jammy) or bake "
            f"the playwright package into a custom image."
        )
    pkg = f"playwright=={pin}"
    logger.info("Browser: installing %s inside container", pkg)
    cmd = ["sh", "-c", f"pip install --quiet --no-input --break-system-packages {pkg}"]
    result = container.exec_run(cmd)
    if result.exit_code != 0:
        cmd = ["sh", "-c", f"pip install --quiet --no-input {pkg}"]
        result = container.exec_run(cmd)
    if result.exit_code != 0:
        out = (result.output or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to install {pkg} in browser container: {out.strip()}")


def _install_micro_server(container) -> None:
    with open(_MICRO_SERVER_PATH_HOST, "rb") as f:
        script_bytes = f.read()
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT)
    info = tarfile.TarInfo(name="restai_browser/micro_server.py")
    info.size = len(script_bytes)
    info.mode = 0o644
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(script_bytes))
    tar.close()
    container.exec_run(["sh", "-c", "mkdir -p /opt"])
    ok = container.put_archive("/opt", buf.getvalue())
    if not ok:
        raise RuntimeError("Browser: failed to put_archive micro_server.py into container")


def _start_micro_server(container) -> None:
    cmd = ["sh", "-c", f"python3 {_MICRO_SERVER_PATH_CONTAINER} > /tmp/browser_server.log 2>&1 &"]
    container.exec_run(cmd, detach=True)


def _wait_healthy(host_port: int) -> None:
    deadline = time.time() + _HEALTH_TIMEOUT
    url = f"http://127.0.0.1:{host_port}/health"
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.3)
    raise RuntimeError(f"Browser: micro-server health check timed out ({last_err})")


def _create_container(scoped_id: str):
    c = _get_client()
    if c is None:
        raise RuntimeError("Browser runtime is not configured")
    image = (getattr(_cfg, "BROWSER_IMAGE", _DEFAULT_IMAGE) or _DEFAULT_IMAGE)
    network = (getattr(_cfg, "BROWSER_NETWORK", "bridge") or "bridge")
    # Opt-in read-only rootfs. Off by default because it is incompatible with
    # the runtime `pip install playwright` step (pip writes to site-packages):
    # enable it only with a custom image that bakes the playwright package in.
    # When on, the writable paths Chromium/Playwright need are backed by tmpfs.
    read_only = bool(getattr(_cfg, "BROWSER_READ_ONLY", False))
    run_kwargs = {}
    if read_only:
        run_kwargs["read_only"] = True
        run_kwargs["tmpfs"] = {"/tmp": "", "/home": "", "/root": "", "/dev/shm": "size=512m"}
    logger.info("Browser: creating container for chat_id=%s", scoped_id)
    from restai.observability.instance import get_instance_id
    container = c.containers.run(
        image,
        command=["sleep", "infinity"],
        detach=True,
        labels={
            "restai.browser_managed": "true",
            "restai.browser_chat_id": scoped_id,
            "restai.created_at": str(int(time.time())),
            "restai.observability.instance_id": get_instance_id(),
        },
        mem_limit="1g",
        cpu_period=100000,
        cpu_quota=100000,
        # Cap process count to blunt fork-bombs from a hostile/exploited page;
        # block setuid privilege escalation inside the container.
        pids_limit=512,
        security_opt=["no-new-privileges"],
        network_mode=network,
        # Chromium needs >64M /dev/shm or tabs crash unpredictably.
        shm_size="512m",
        ports={f"{_CONTAINER_PORT}/tcp": ("127.0.0.1", None)},
        remove=True,
        **run_kwargs,
    )
    container.reload()
    port = _discover_port(container)
    if port is None:
        try:
            container.stop(timeout=3)
        except Exception:
            pass
        raise RuntimeError("Browser: Docker did not publish a host port")
    _install_micro_server(container)
    _ensure_playwright_pkg(container, image)
    _start_micro_server(container)
    _wait_healthy(port)
    return container, port


def _get_or_create(scoped_id: str):
    container = _resolve_container(scoped_id)
    if container is not None:
        port = _discover_port(container)
        if port is not None:
            return container, port
    return _create_container(scoped_id)


def _chat_lock(scoped_id: str) -> asyncio.Lock:
    lock = _chat_locks.get(scoped_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_locks[scoped_id] = lock
    return lock


def _remove_by_scoped_id(scoped_id: str) -> None:
    """Stop the container identified by an already-scoped id. Idempotent."""
    container = _resolve_container(scoped_id)
    _drop_db_activity(scoped_id)
    if container is None:
        return
    try:
        container.stop(timeout=3)
        logger.info("Browser: removed container for chat_id=%s", scoped_id)
    except Exception as e:
        logger.warning("Browser: failed to stop container for chat_id=%s: %s", scoped_id, e)


def remove_container(chat_id: str, project_id=None) -> None:
    """Stop the per-chat container if it exists. Idempotent."""
    _remove_by_scoped_id(_scoped_id(project_id, chat_id))


def _touch_db_activity(chat_id: str, container_id: Optional[str]) -> None:
    if not chat_id:
        return
    try:
        from restai.database import open_db_wrapper
        db = open_db_wrapper()
        try:
            db.upsert_browser_activity(chat_id, container_id)
        finally:
            db.db.close()
    except Exception as e:
        logger.debug("browser_chat_activity upsert failed for %s: %s", chat_id, e)


def _drop_db_activity(chat_id: str) -> None:
    if not chat_id:
        return
    try:
        from restai.database import open_db_wrapper
        db = open_db_wrapper()
        try:
            db.delete_browser_activity(chat_id)
        finally:
            db.db.close()
    except Exception as e:
        logger.debug("browser_chat_activity delete failed for %s: %s", chat_id, e)


def call(chat_id: str, path: str, payload: Optional[dict] = None, project_id=None) -> dict:
    """Post JSON to the in-container micro-server and return parsed response."""
    chat_id = _scoped_id(project_id, chat_id)
    container, port = _get_or_create(chat_id)
    _touch_db_activity(chat_id, container.id)

    url = f"http://127.0.0.1:{port}{path}"
    try:
        resp = requests.post(url, json=payload or {}, timeout=90)
    except requests.exceptions.RequestException:
        # Container might have died; drop + retry once. `chat_id` is already
        # the scoped id here, so use the scoped-id remover directly to keep
        # the label/lookup consistent (re-scoping would double-prefix).
        _remove_by_scoped_id(chat_id)
        container, port = _get_or_create(chat_id)
        _touch_db_activity(chat_id, container.id)
        url = f"http://127.0.0.1:{port}{path}"
        resp = requests.post(url, json=payload or {}, timeout=90)

    # Re-touch after request returns so a long browser call (e.g.
    # navigation that takes >timeout) doesn't leave a stale heartbeat
    # that the next cron tick would evict on.
    _touch_db_activity(chat_id, container.id)

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"browser {path}: {detail}")
    return resp.json()
