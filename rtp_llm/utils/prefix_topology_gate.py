import asyncio
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager

from rtp_llm.utils.parallel_topology import write_runtime_dp_addresses


class PrefixTopologyConflict(Exception):
    pass


class PrefixTopologyUnavailable(Exception):
    pass


class PrefixTopologyGate:
    def __init__(self, path, addresses, tp_size, dp_size):
        self.path = path + ".prefix-control"
        self.mutex = self.path + ".lock"
        self.readers = self.path + ".readers"
        with open(self.mutex, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not os.path.exists(self.path):
                write_runtime_dp_addresses(path, addresses, tp_size, dp_size)
                self._write(None)
            self.pending()

    def pending(self):
        with open(self.path) as stream:
            state = json.load(stream)
        if not isinstance(state, dict) or set(state) != {"pending"}:
            raise ValueError("invalid shared topology state")
        pending = state["pending"]
        if pending is not None and not isinstance(pending, dict):
            raise ValueError("invalid shared topology identity")
        return pending

    def _write(self, identity):
        temporary = self.path + f".{os.getpid()}.tmp"
        with open(temporary, "w") as stream:
            json.dump({"pending": identity}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        descriptor = os.open(os.path.dirname(self.path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @asynccontextmanager
    async def _exclusive(self, path, deadline):
        with open(path, "a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "shared frontend topology drain is unconfirmed"
                        )
                    await asyncio.sleep(0.005)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @asynccontextmanager
    async def switch(self, identity, timeout_ms):
        deadline = time.monotonic() + timeout_ms / 1000
        async with self._exclusive(self.mutex, deadline):
            if self.pending() not in (None, identity):
                raise PrefixTopologyConflict(
                    "another topology plan remains unconfirmed"
                )
            self._write(identity)
            async with self._exclusive(self.readers, deadline):
                yield

    def finish(self, identity):
        if self.pending() != identity:
            raise PrefixTopologyConflict("shared topology identity changed")
        self._write(None)

    @asynccontextmanager
    async def admission(self):
        with open(self.readers, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                if self.pending() is not None:
                    raise PrefixTopologyUnavailable()
            except (OSError, ValueError) as error:
                raise PrefixTopologyUnavailable() from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class PrefixTopologyMiddleware:
    def __init__(self, app, gate, refresh):
        self.app = app
        self.gate = gate
        self.refresh = refresh

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        inference = scope.get("method") == "POST" and (
            path
            in {
                "/",
                "/chat/completions",
                "/v1/chat/completions",
                "/v1/batch/chat/completions",
                "/batch_infer",
            }
            or path.startswith("/v1/embeddings")
            or path in {"/v1/reranker", "/v1/classifier"}
        )
        if scope["type"] != "http" or not inference:
            await self.app(scope, receive, send)
            return
        try:
            async with self.gate.admission():
                await self.refresh()
                await self.app(scope, receive, send)
        except PrefixTopologyUnavailable:
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error":"topology_unconfirmed"}',
                }
            )
