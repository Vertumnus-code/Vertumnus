import asyncio
import hmac
import json
import os
import re
import time
import uuid
from urllib.parse import urlsplit

import aiohttp


class PrefixTopologyWriterClient:
    def __init__(self, url, token, route):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token)
            or not route
        ):
            raise ValueError("invalid prefix topology writer configuration")
        self.url = url.rstrip("/") + "/rtp_llm/prefix_topology/plans"
        self._authorization = "Bearer " + token
        self.route = route

    @classmethod
    def from_environment(cls):
        url = os.environ.get("PREFIX_TOPOLOGY_WRITER_URL", "")
        token = os.environ.get("PREFIX_TOPOLOGY_WRITER_TOKEN", "")
        route = os.environ.get("PREFIX_TOPOLOGY_FRONTEND_ROUTE", "")
        if not url and not token and not route:
            return None
        return cls(url, token, route)

    def authorized(self, authorization):
        return isinstance(authorization, str) and hmac.compare_digest(
            self._authorization.encode(), authorization.encode()
        )

    def validate(self, request):
        allowed = {
            "action",
            "switch_id",
            "tp_size",
            "dp_size",
            "timeout_ms",
            "deadline_unix_ms",
        }
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("managed topology requires an immutable writer request")
        for key in ["tp_size", "dp_size", "deadline_unix_ms"]:
            if type(request.get(key)) is not int or request[key] <= 0:
                raise ValueError("managed topology requires positive integer bounds")
        timeout = request.get("timeout_ms", 300000)
        if type(timeout) is not int or not 0 < timeout <= 300000:
            raise ValueError("managed topology requires a bounded observation timeout")
        if (
            request.get("action", "switch_parallel_topology")
            != "switch_parallel_topology"
            or request["tp_size"] * request["dp_size"] > 4096
            or str(uuid.UUID(request.get("switch_id", ""))) != request["switch_id"]
            or request["deadline_unix_ms"] - int(time.time() * 1000) > 300000
        ):
            raise ValueError("invalid immutable topology identity or deadline")
        return {
            "switchId": request["switch_id"],
            "route": self.route,
            "tpSize": request["tp_size"],
            "dpSize": request["dp_size"],
            "deadlineUnixMs": request["deadline_unix_ms"],
        }

    async def _request(self, method, suffix, deadline, body=None):
        while True:
            try:
                return await self._request_once(method, suffix, deadline, body)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if time.time() * 1000 >= deadline:
                    raise TimeoutError(
                        "topology outcome remains unconfirmed at caller deadline"
                    )
                # Every endpoint binds an immutable plan, so an ambiguous reply
                # can be retried without creating another switch or extending it.
                await asyncio.sleep(min(0.1, (deadline - time.time() * 1000) / 1000))

    async def _request_once(self, method, suffix, deadline, body=None):
        remaining = (deadline - time.time() * 1000) / 1000
        if remaining <= 0:
            raise TimeoutError(
                "topology outcome remains unconfirmed at caller deadline"
            )
        # Advancing can copy several long prefixes. A short observation timeout
        # cancels the server task halfway through that durable plan.
        total = (
            remaining
            if method == "POST" and suffix.endswith("/advance")
            else min(remaining, 30)
        )
        timeout = aiohttp.ClientTimeout(total=total, connect=min(remaining, 30))
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            async with session.request(
                method,
                self.url + suffix,
                json=body,
                headers={"Authorization": self._authorization},
                allow_redirects=False,
            ) as response:
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > 16 * 1024 * 1024:
                        raise ValueError("topology response exceeds bound")
                if response.status == 409:
                    return None
                if response.status in {500, 502, 503, 504}:
                    raise aiohttp.ClientError(
                        f"topology writer temporarily unavailable: HTTP {response.status}"
                    )
                if response.status != 200:
                    raise RuntimeError(
                        f"topology writer rejected request: HTTP {response.status}"
                    )
                return json.loads(data)

    def _check(self, response, request):
        if response is None:
            return None
        entry = response["entry"]
        plan = entry["plan"]
        expected = self.validate(request)
        for key in ["switchId", "tpSize", "dpSize", "deadlineUnixMs"]:
            if plan[key] != expected[key]:
                raise ValueError("writer returned a different immutable topology plan")
        return entry["state"]

    async def switch(self, request):
        body = self.validate(request)
        deadline = int(time.time() * 1000) + request.get("timeout_ms", 300000)
        response = await self._request("POST", "", deadline, body)
        while True:
            state = self._check(response, request)
            if state == "CANCELLED":
                return {
                    "status": "cancelled",
                    "managed_topology": True,
                    "switch_id": request["switch_id"],
                }
            if state in {"CONFIRMED", "ADMITTED"}:
                return {
                    "status": "confirmed",
                    "managed_topology": True,
                    "switch_id": request["switch_id"],
                }
            response = await self._request(
                "POST", f"/{request['switch_id']}/advance", deadline
            )
            await asyncio.sleep(0.1)

    async def admit(self, request):
        self.validate(request)
        deadline = int(time.time() * 1000) + request.get("timeout_ms", 300000)
        while True:
            response = await self._request(
                "POST", f"/{request['switch_id']}/admit", deadline
            )
            if (
                self._check(response, request) == "ADMITTED"
                and response["frontendAdmitted"]
            ):
                return {
                    "status": "ok",
                    "managed_topology": True,
                    "switch_id": request["switch_id"],
                }
            await asyncio.sleep(0.1)
