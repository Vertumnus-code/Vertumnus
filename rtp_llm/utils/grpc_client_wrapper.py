import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import grpc
from google.protobuf.json_format import MessageToDict

import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 as pb2
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc as pb2_grpc
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.utils.parallel_topology import compute_dp_addresses
from rtp_llm.utils.prefix_topology_writer import PrefixTopologyWriterClient
from rtp_llm.utils.time_util import Timer


class GrpcClientWrapper:
    """Wrapper for direct gRPC calls to replace async_request_server"""

    def __init__(
        self,
        server_port: int,
        dp_addresses: Optional[List[str]] = None,
        all_rank_addresses: Optional[List[str]] = None,
    ):
        self.server_port = server_port
        self.address = f"localhost:{server_port}"
        self.channel = None
        self.stub = None
        # All DP addresses for broadcast operations (defaults to local address)
        self.dp_addresses = dp_addresses if dp_addresses else [self.address]
        self.all_rank_addresses = (
            all_rank_addresses if all_rank_addresses else list(self.dp_addresses)
        )
        self._dp_channels: Dict[str, Any] = {}
        self._dp_stubs: Dict[str, Any] = {}
        self.topology_writer = PrefixTopologyWriterClient.from_environment()

    async def update_parallel_topology_addresses(
        self, tp_size: int, dp_size: int
    ) -> List[str]:
        """Update frontend-side DP root addresses after a hot topology switch."""
        new_addresses = compute_dp_addresses(
            self.all_rank_addresses, int(tp_size), int(dp_size)
        )
        old_addresses = set(self.dp_addresses)
        self.dp_addresses = new_addresses
        for address in old_addresses - set(new_addresses):
            channel = self._dp_channels.pop(address, None)
            self._dp_stubs.pop(address, None)
            if channel is not None:
                try:
                    await channel.close()
                except Exception as e:
                    logging.warning(f"Failed to close stale DP channel {address}: {e}")
        logging.info("updated grpc client DP addresses: %s", self.dp_addresses)
        return self.dp_addresses

    async def _ensure_connection(self):
        """Ensure gRPC channel and stub are created"""
        if self.channel is None or self.stub is None:
            self.channel = grpc.aio.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self.stub = RpcServiceStub(self.channel)

    async def _ensure_dp_connection(self, address: str):
        """Ensure gRPC channel and stub are created for a specific DP address"""
        if address not in self._dp_channels or self._dp_stubs.get(address) is None:
            self._dp_channels[address] = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.max_metadata_size", 1024 * 1024 * 1024),
                ],
            )
            self._dp_stubs[address] = RpcServiceStub(self._dp_channels[address])

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
        for address, channel in self._dp_channels.items():
            try:
                await channel.close()
            except Exception as e:
                logging.warning(f"Failed to close DP channel for {address}: {e}")
        self._dp_channels.clear()
        self._dp_stubs.clear()

    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            await self._ensure_connection()
            # Using a simple request to check if server is responsive
            request = pb2.EmptyPB()
            await self.stub.CheckHealth(request, timeout=1)
            return {"status": "ok"}
        except Exception as e:
            return {
                "status": "error",
                "message": e,
            }

    async def get_cache_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cache status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.CacheVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            need_cache_keys=query_params.get("need_cache_keys", True),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetCacheStatus(request, timeout=1)
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get cache status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.CACHE_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.CACHE_STATUS_QPS_LATENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for cache_status"}
        # For backward compatibility, merge first result's fields at top level
        merged = dict(results[0])
        merged["results"] = results
        merged["dp_size"] = len(results)
        return merged

    async def get_worker_status(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get worker status from all DP backends.

        Returns a list of per-DP results under the ``"results"`` key.
        If no backend is reachable, returns ``{"error": ...}``.
        """
        start_time = time.time() * 1000
        request = pb2.StatusVersionPB(
            latest_cache_version=query_params.get("latest_cache_version", -1),
            latest_finished_version=query_params.get("latest_finished_version", -1),
        )
        results = []
        for addr in self.dp_addresses:
            try:
                await self._ensure_dp_connection(addr)
                response = await self._dp_stubs[addr].GetWorkerStatus(
                    request, timeout=1
                )
                result = MessageToDict(
                    response,
                    preserving_proto_field_name=True,
                    including_default_value_fields=True,
                )
                result["address"] = addr
                results.append(result)
            except Exception as e:
                logging.warning(f"Get worker status from {addr} failed: {e}")
        kmonitor.report(AccMetrics.WORKER_STATUS_QPS_METRIC, 1)
        kmonitor.report(
            GaugeMetrics.WORKER_STATUS_QPS_LANTENCY_METRIC,
            time.time() * 1000 - start_time,
        )
        if not results:
            return {"error": "No backend available for worker_status"}
        merged = dict(results[0])
        merged["results"] = results
        return merged

    async def set_log_level(self, req: Any) -> Dict[str, Any]:
        """Set log level - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            request = pb2.SetLogLevelRequestPB(
                log_level=req.get("log_level", "INFO"),
            )
            await self.stub.SetLogLevel(request, timeout=3)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Set log level failed: {e}")
            return {"error": f"Failed to set log level: {str(e)}"}

    async def start_profile(self, req: Any) -> Dict[str, Any]:
        """Start profiling switch in backend process"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            if req is None:
                req = {}
            request = pb2.StartProfileRequestPB(
                trace_name=str(req.get("trace_name", "")),
                start_step=int(req.get("start_step", 0)),
                num_steps=int(req.get("num_steps", 0)),
                enable_all_rank=bool(
                    req.get("enable_all_rank", req.get("all_tp", False))
                ),
            )
            await self.stub.StartProfile(request, timeout=3)
            return {"status": "ok"}

        except Exception as e:
            logging.error(f"Start profile failed: {e}")
            return {"error": f"Failed to start profile: {str(e)}"}

    async def update_eplb_config(self, req: Any) -> Dict[str, Any]:
        """Update EPLB config - this would need to be implemented based on your requirements"""
        try:
            await self._ensure_connection()
            if isinstance(req, str):
                req = json.loads(req)
            epld_req = pb2.UpdateEplbConfigRequestPB(
                mode=req.get("mode", "NONE"),
                update_time=int(time.time()),
            )
            await self.stub.UpdateEplbConfig(epld_req)
            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Update EPLB config failed: {e}")
            return {"error": f"Failed to update EPLB config: {str(e)}"}

    async def update_scheduler_info(self, req: Any) -> Dict[str, Any]:
        """Update scheduler info on all DP addresses"""
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if (
                isinstance(req, dict)
                and req.get("action") == "switch_parallel_topology"
            ):
                return await self.switch_parallel_topology(req)
            update_schedule_info_req = pb2.UpdateSchedulerInfoRequestPB(
                scheduler_info=json.dumps(req)
            )

            async def send_to_address(address: str):
                await self._ensure_dp_connection(address)
                await self._dp_stubs[address].UpdateSchedulerInfo(
                    update_schedule_info_req
                )

            tasks = [send_to_address(addr) for addr in self.dp_addresses]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors = [
                f"{self.dp_addresses[i]}: {str(r)}"
                for i, r in enumerate(results)
                if isinstance(r, Exception)
            ]
            if errors:
                logging.error(
                    f"Update scheduler info failed on some addresses: {errors}"
                )
                return {"error": f"Failed on some addresses: {errors}"}

            return {"status": "ok"}
        except Exception as e:
            logging.error(f"Update scheduler info failed: {e}")
            return {"error": f"Failed to update scheduler info: {str(e)}"}

    async def switch_parallel_topology(self, req: Any) -> Dict[str, Any]:
        """Submit the caller's immutable topology plan to world rank zero only."""
        if isinstance(req, str):
            req = json.loads(req)
        req = dict(req or {})
        if req.get("internal", False) or req.get("phase", "commit") != "commit":
            raise ValueError("frontend topology requests cannot invoke internal phases")
        timeout_ms = req.get("timeout_ms", 300000)
        if (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or not 0 < timeout_ms <= 300000
        ):
            raise ValueError("topology timeout_ms must be an integer in [1, 300000]")
        req["action"] = "switch_parallel_topology"
        if self.topology_writer is not None:
            return await self.topology_writer.switch(req)
        address = self.all_rank_addresses[0]
        await self._ensure_dp_connection(address)
        await self._dp_stubs[address].UpdateSchedulerInfo(
            pb2.UpdateSchedulerInfoRequestPB(scheduler_info=json.dumps(req)),
            timeout=timeout_ms / 1000.0,
        )
        return {"status": "ok"}

    async def post_request(self, uri: str, req: Dict[str, Any]) -> Dict[str, Any]:
        """Generic POST request handler - routes to appropriate method based on URI"""
        try:
            if uri == "health_check":
                return await self.health_check()
            elif uri == "cache_status":
                return await self.get_cache_status(req)
            elif uri == "worker_status":
                return await self.get_worker_status(req)
            elif uri == "set_log_level":
                return await self.set_log_level(req)
            elif uri == "start_profile":
                return await self.start_profile(req)
            elif uri == "update_eplb_config":
                return await self.update_eplb_config(req)
            elif uri == "update_scheduler_info":
                return await self.update_scheduler_info(req)
            elif uri == "switch_parallel_topology":
                return await self.switch_parallel_topology(req)
            else:
                # Default case - return empty success
                return {"status": "ok"}
        except Exception as e:
            logging.error(f"POST request to {uri} failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
