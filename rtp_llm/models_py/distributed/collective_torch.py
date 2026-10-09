from __future__ import annotations

import gc
import logging
import os
import threading
import time
import weakref
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
import torch.distributed

from rtp_llm.models_py.distributed import rocm_rccl
from rtp_llm.models_py.distributed.symm_mem import (
    clear_symm_mem_communicator,
    get_symm_mem_communicator,
    init_symm_mem_communicator,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig

# ParallelMode enum values matching C++ rtp_llm::ParallelMode in OpData.h
_CPP_PARALLEL_MODE_TP = 0
_CPP_PARALLEL_MODE_DP = 1
_CPP_PARALLEL_MODE_DP_AND_TP = 2


class Group(Enum):
    """Process group types for collective operations"""

    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


# Global process group storage
# Key can be Group enum or string (for multiple DP/TP groups)
_group_map: Dict[Union[Group, str], torch.distributed.ProcessGroup] = {}
_cpu_group_map: Dict[Union[Group, str], torch.distributed.ProcessGroup] = {}
_parallelism_config: Optional[ParallelismConfig] = None
_initialized: bool = False  # Track if we've initialized (to prevent double init)
_parallelism_config_modules: weakref.WeakSet[torch.nn.Module] = weakref.WeakSet()
_comm_ops_lock = threading.RLock()
_comm_ops_callbacks = None
_resident_groups = {}
_resident_topologies = {}


def _new_topology_group(ranks, backend, timeout):
    if not os.environ.get("RTP_LLM_PREWARM_TOPOLOGIES", "").strip():
        return torch.distributed.new_group(
            ranks=ranks, backend=backend, timeout=timeout
        )
    key = (backend, tuple(ranks))
    if key not in _resident_groups:
        _resident_groups[key] = torch.distributed.new_group(
            ranks=ranks, backend=backend, timeout=timeout
        )
    return _resident_groups[key]


def register_parallelism_config_module(module: torch.nn.Module) -> None:
    _parallelism_config_modules.add(module)


def init_distributed_environment(
    parallelism_config: ParallelismConfig,
    nccl_comm_config: NcclCommConfig,
    nccl_init_port: int,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    """Initialize distributed environment and create process groups.

    This function creates DP, TP, and DP_AND_TP process groups using torch.distributed.
    It can only be called once unless destroy_distributed_environment() has been called.

    Args:
        parallelism_config: Configuration for parallelism setup (sizes, ranks, etc.)
        nccl_comm_config: NCCL config with nccl_ip (and other ports for C++ init).
        nccl_init_port: Port for torch.distributed init_process_group (tcp://ip:port).
        backend: Distributed backend (default: "nccl")
        timeout: Timeout in seconds for process group initialization

    Raises:
        RuntimeError: If already initialized and not destroyed
    """
    global _group_map, _cpu_group_map, _parallelism_config, _initialized

    # Check if already initialized (and not destroyed)
    if _initialized and torch.distributed.is_initialized():
        logging.warning(
            "Distributed environment already initialized, skipping initialization"
        )
        # Still need to create groups if they don't exist
        if not _group_map:
            _create_process_groups(parallelism_config, backend, timedelta(days=36500))
            _register_process_groups_to_cpp()
        if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        return

    assert backend in ["nccl"], "backend current only supports nccl"
    ip = nccl_comm_config.nccl_ip
    port = nccl_init_port
    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    local_rank = parallelism_config.local_rank

    rocm_rccl.configure_process_groups(parallelism_config)
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"

    # If torch.distributed is already initialized (e.g., by external code),
    # we still need to create our process groups
    if torch.distributed.is_initialized():
        logging.info("torch.distributed already initialized, creating process groups")
        if Group.DP_AND_TP not in _cpu_group_map:
            _cpu_group_map[Group.DP_AND_TP] = torch.distributed.new_group(
                ranks=list(range(parallelism_config.world_size)),
                backend="gloo",
                timeout=timedelta(days=36500),
            )
        _create_process_groups(parallelism_config, backend, timedelta(days=36500))
        _parallelism_config = parallelism_config
        _initialized = True
        _register_process_groups_to_cpp()
        if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
            rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
        return

    logging.info(
        f"[rank: {world_rank}] initialize process_group: {ip}:{port}, rank: {world_rank}, world_size: {world_size}, "
        f"local_rank: {local_rank}, backend: {backend}, timeout: {timeout}",
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Use a very large timeout for NCCL so that workers simply block
    # until rank 0 has real work, instead of crashing with a timeout.
    # Note: timedelta.max overflows in PyTorch's C++ TCP store, so use 100 years instead.
    infinite_timeout = timedelta(days=36500)

    # DP_AND_TP (global group) - initialized via init_process_group
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=world_rank,
        # device_id=torch.device(f"cuda:{local_rank}"), # https://github.com/pytorch/pytorch/pull/149144
        timeout=infinite_timeout,
    )
    torch.distributed.barrier(group=torch.distributed.group.WORLD)
    _group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    _cpu_group_map[Group.DP_AND_TP] = torch.distributed.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=infinite_timeout,
    )
    logging.info(
        f"[rank: {world_rank}] Created DP_AND_TP group {torch.distributed.group.WORLD} with ranks: {list(range(world_size))}"
    )

    # Create DP and TP groups
    _create_process_groups(parallelism_config, backend, timedelta(days=36500))
    _parallelism_config = parallelism_config
    _initialized = True
    _register_process_groups_to_cpp()
    if rocm_rccl.is_available_runtime() and parallelism_config.tp_size > 1:
        rocm_rccl.prepare_comm_if_needed(parallelism_config, _get_group(Group.TP))
    init_user_buffers_environment(parallelism_config)
    prewarm_parallel_topologies_from_env()


def _create_process_groups(
    parallelism_config: ParallelismConfig,
    backend: str,
    timeout: Optional[timedelta],
):
    """Create DP and TP process groups.

    Args:
        parallelism_config: Configuration for parallelism setup
        backend: Distributed backend
        timeout: Timeout for process group creation
    """
    global _group_map, _cpu_group_map

    world_rank = parallelism_config.world_rank
    world_size = parallelism_config.world_size
    tp_size = parallelism_config.tp_size
    dp_size = parallelism_config.dp_size
    begin = time.monotonic()

    _group_map = {
        key: group for key, group in _group_map.items() if key == Group.DP_AND_TP
    }
    _cpu_group_map = {
        key: group for key, group in _cpu_group_map.items() if key == Group.DP_AND_TP
    }

    if world_size != dp_size:
        # Create all DP groups - all ranks must participate in creating all DP groups
        # DP group: ranks with the same tp_rank (i.e., world_rank % tp_size)
        # There are tp_size DP groups (one for each tp_rank value)
        for tp_rank_val in range(tp_size):
            dp_ranks = [r for r in range(world_size) if r % tp_size == tp_rank_val]
            if len(dp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating DP group for tp_rank {tp_rank_val} with ranks: {dp_ranks}"
                )
                dp_group = _new_topology_group(
                    ranks=dp_ranks,
                    backend=backend,
                    timeout=timedelta(days=36500),
                )
                dp_cpu_group = _new_topology_group(
                    ranks=dp_ranks,
                    backend="gloo",
                    timeout=timedelta(days=36500),
                )
                # Only store the group if this rank is part of it
                if world_rank in dp_ranks:
                    group_key = Group.DP.name + str(tp_rank_val)
                    _group_map[group_key] = dp_group
                    _cpu_group_map[group_key] = dp_cpu_group
                    logging.info(
                        f"[rank: {world_rank}] Stored DP group with key: {group_key} {dp_group} with ranks: {dp_ranks}"
                    )
                # All ranks must wait for group creation to complete
                torch.distributed.barrier()

    if world_size != tp_size:
        # Create all TP groups - all ranks must participate in creating all TP groups
        # TP group: ranks with the same dp_rank (i.e., world_rank // tp_size)
        # There are dp_size TP groups (one for each dp_rank value)
        for dp_rank_val in range(dp_size):
            tp_ranks = [r for r in range(world_size) if r // tp_size == dp_rank_val]
            if len(tp_ranks) > 0:
                logging.info(
                    f"[rank: {world_rank}] Creating TP group for dp_rank {dp_rank_val} with ranks: {tp_ranks}"
                )
                tp_group = _new_topology_group(
                    ranks=tp_ranks,
                    backend=backend,
                    timeout=timedelta(days=36500),
                )
                tp_cpu_group = _new_topology_group(
                    ranks=tp_ranks,
                    backend="gloo",
                    timeout=timedelta(days=36500),
                )
                # Only store the group if this rank is part of it
                if world_rank in tp_ranks:
                    group_key = Group.TP.name + str(dp_rank_val)
                    _group_map[group_key] = tp_group
                    _cpu_group_map[group_key] = tp_cpu_group
                    logging.info(
                        f"[rank: {world_rank}] Stored TP group with key: {group_key} {tp_group} with ranks: {tp_ranks}"
                    )

                if world_rank in tp_ranks and tp_size > 1:
                    init_symm_mem_communicator(tp_group)

                # All ranks must wait for group creation to complete
                torch.distributed.barrier()
    elif tp_size > 1 and world_size == tp_size:
        # Single TP group: WORLD is the TP group, init symm_mem for it
        init_symm_mem_communicator(torch.distributed.group.WORLD)

    logging.info(
        f"[rank: {world_rank}] Created process groups for TP{tp_size}DP{dp_size}, "
        f"cost_ms={(time.monotonic() - begin) * 1000:.2f}"
    )


def _destroy_active_topology_process_groups() -> None:
    """Destroy non-WORLD groups for the currently active TP/DP topology.

    Hot switching is intentionally single-topology: model workers, weights, KV
    storage, and the WORLD coordination group stay alive, while topology-specific
    NCCL/Gloo groups are released before the replacement topology is created.
    """
    global _group_map, _cpu_group_map

    world_rank = torch.distributed.get_rank()
    groups = []
    seen: set[int] = set()
    for group_map in (_group_map, _cpu_group_map):
        for key, process_group in sorted(
            group_map.items(), key=lambda item: str(item[0])
        ):
            if key == Group.DP_AND_TP or process_group is None:
                continue
            group_id = id(process_group)
            if group_id in seen:
                continue
            seen.add(group_id)
            groups.append(process_group)

    for process_group in groups:
        torch.distributed.destroy_process_group(process_group)

    _group_map = {
        key: group for key, group in _group_map.items() if key == Group.DP_AND_TP
    }
    _cpu_group_map = {
        key: group for key, group in _cpu_group_map.items() if key == Group.DP_AND_TP
    }
    gc.collect()
    logging.info(
        f"[rank: {world_rank}] destroyed {len(groups)} active topology process groups"
    )


def _register_process_groups_to_cpp():
    """Register Python comm op callbacks for C++ to call back into."""
    global _comm_ops_callbacks
    try:
        import librtp_compute_ops

        if not hasattr(librtp_compute_ops, "register_comm_ops"):
            logging.debug(
                "register_comm_ops not available, skip C++ comm ops registration"
            )
            return
    except ImportError:
        logging.debug(
            "librtp_compute_ops not available, skip C++ comm ops registration"
        )
        return

    # Build mode -> process_group mapping (int mode -> ProcessGroup)
    mode_to_group: Dict[int, torch.distributed.ProcessGroup] = {}
    mode_to_cpu_group: Dict[int, torch.distributed.ProcessGroup] = {}
    registered_modes: set = set()

    def register_mode(
        mode: int,
        group_key: Union[Group, str],
        pg: torch.distributed.ProcessGroup,
    ) -> None:
        mode_to_group[mode] = pg
        cpu_pg = _cpu_group_map.get(group_key)
        if cpu_pg is not None:
            mode_to_cpu_group[mode] = cpu_pg

    for group_key, pg in _group_map.items():
        if group_key == Group.DP_AND_TP:
            if _CPP_PARALLEL_MODE_DP_AND_TP not in registered_modes:
                register_mode(_CPP_PARALLEL_MODE_DP_AND_TP, group_key, pg)
                registered_modes.add(_CPP_PARALLEL_MODE_DP_AND_TP)
        elif isinstance(group_key, str):
            if group_key.startswith(Group.TP.name):
                if _parallelism_config is not None:
                    if _parallelism_config.world_size == _parallelism_config.tp_size:
                        continue
                    dp_rank = (
                        torch.distributed.get_rank() // _parallelism_config.tp_size
                    )
                    expected_key = Group.TP.name + str(dp_rank)
                    if (
                        group_key == expected_key
                        and _CPP_PARALLEL_MODE_TP not in registered_modes
                    ):
                        register_mode(_CPP_PARALLEL_MODE_TP, group_key, pg)
                        registered_modes.add(_CPP_PARALLEL_MODE_TP)
            elif group_key.startswith(Group.DP.name):
                if _parallelism_config is not None:
                    if _parallelism_config.dp_size <= 1:
                        continue
                    tp_rank = torch.distributed.get_rank() % _parallelism_config.tp_size
                    expected_key = Group.DP.name + str(tp_rank)
                    if (
                        group_key == expected_key
                        and _CPP_PARALLEL_MODE_DP not in registered_modes
                    ):
                        register_mode(_CPP_PARALLEL_MODE_DP, group_key, pg)
                        registered_modes.add(_CPP_PARALLEL_MODE_DP)

    # If world_size == tp_size, WORLD is also TP group.
    if (
        _parallelism_config is not None
        and _parallelism_config.tp_size > 1
        and _parallelism_config.world_size == _parallelism_config.tp_size
        and _CPP_PARALLEL_MODE_TP not in registered_modes
    ):
        pg_world = _group_map.get(Group.DP_AND_TP)
        if pg_world is not None:
            register_mode(_CPP_PARALLEL_MODE_TP, Group.DP_AND_TP, pg_world)

    # NOTE: These callbacks are NOT thin wrappers around the module-level broadcast()/
    # all_reduce()/all_gather() because the C++ calling convention differs significantly:
    #   - C++ uses int mode (ParallelMode enum ordinal) instead of Group enum
    #   - execBroadcast passes multiple tensors + CPU tensors needing GPU promotion
    #   - execAllReduce supports dest tensor + multiple ReduceOp types
    #   - execAllGather writes into pre-allocated recv_buffers with inplace mode
    # The module-level functions have different signatures and semantics (e.g. all_gather
    # allocates a new tensor), so we implement the C++ contract directly here.

    def _callback_cuda_device_id() -> int:
        if _parallelism_config is not None:
            return int(_parallelism_config.local_rank)
        return torch.cuda.current_device()

    def _ensure_cuda(t: torch.Tensor, device_id: int):
        """Move CPU tensor to CUDA if needed (NCCL requires CUDA tensors)."""
        if t.is_cuda:
            return t, False
        return t.to(torch.device("cuda", device_id)), True

    def cpp_broadcast(tensors: List[torch.Tensor], root: int, mode: int) -> None:
        """Broadcast tensors from root rank to all ranks in the group.

        Args:
            tensors: Tensors to broadcast, each is broadcast in-place from root.
            root: Source rank that holds the data.
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return
        try:
            cpu_pg = mode_to_cpu_group.get(mode)
            global_root = torch.distributed.get_global_rank(pg, root)
            device_id = _callback_cuda_device_id()
            for t in tensors:
                if not t.is_cuda and cpu_pg is not None:
                    cpu_global_root = torch.distributed.get_global_rank(cpu_pg, root)
                    torch.distributed.broadcast(t, cpu_global_root, group=cpu_pg)
                    continue
                gpu_t, was_cpu = _ensure_cuda(t, device_id)
                torch.distributed.broadcast(gpu_t, global_root, group=pg)
                if was_cpu:
                    t.copy_(gpu_t)
        except Exception:
            tensor_desc = [
                {
                    "shape": tuple(t.shape),
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                    "numel": t.numel(),
                }
                for t in tensors
            ]
            logging.exception(
                "cpp_broadcast failed: rank=%s mode=%s root=%s group_size=%s tensors=%s",
                torch.distributed.get_rank(),
                mode,
                root,
                pg.size(),
                tensor_desc,
            )
            raise

    _REDUCE_OPS = {
        0: torch.distributed.ReduceOp.SUM,
        1: torch.distributed.ReduceOp.PRODUCT,
        2: torch.distributed.ReduceOp.MAX,
        3: torch.distributed.ReduceOp.MIN,
        4: torch.distributed.ReduceOp.AVG,
    }

    def cpp_allreduce(
        tensor: torch.Tensor, op: int, mode: int, dest: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """All-reduce a tensor across ranks in the group.

        Args:
            tensor: Input tensor to reduce.
            op: ReduceOp int (0=SUM, 1=PROD, 2=MAX, 3=MIN, 4=AVG).
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
            dest: If not None, result is written here instead of reducing in-place on tensor.
        Returns:
            The reduced tensor (dest if provided, otherwise tensor).
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return tensor if dest is None else tensor
        target = dest if dest is not None else tensor
        if dest is not None:
            target.copy_(tensor)
        cpu_pg = mode_to_cpu_group.get(mode)
        if not target.is_cuda and cpu_pg is not None:
            torch.distributed.all_reduce(
                target,
                op=_REDUCE_OPS.get(op, torch.distributed.ReduceOp.SUM),
                group=cpu_pg,
            )
            return target
        device_id = _callback_cuda_device_id()
        gpu_t, was_cpu = _ensure_cuda(target, device_id)
        torch.distributed.all_reduce(
            gpu_t, op=_REDUCE_OPS.get(op, torch.distributed.ReduceOp.SUM), group=pg
        )
        if was_cpu:
            target.copy_(gpu_t)
        return target

    def cpp_allgather(
        recv_buffers: List[torch.Tensor],
        mode: int,
        send_buffers: List[torch.Tensor],
        inplace: bool,
    ) -> None:
        """All-gather tensors from all ranks into recv_buffers.

        Args:
            recv_buffers: Output tensors, each of size [world_size * per_rank_numel].
            mode: ParallelMode int (0=TP, 1=DP, 2=DP_AND_TP) selecting process group.
            send_buffers: Per-rank input tensors (used when inplace=False).
            inplace: If True, each rank's send data is extracted from its slice in recv_buffers;
                     if False, send data comes from send_buffers.
        """
        pg = mode_to_group.get(mode)
        if pg is None or pg.size() < 2:
            return
        cpu_pg = mode_to_cpu_group.get(mode)
        device_id = _callback_cuda_device_id()
        rank = pg.rank()
        world_size = pg.size()
        for i, recv_buf in enumerate(recv_buffers):
            data_num = recv_buf.numel() // world_size
            recv_on_cpu = not recv_buf.is_cuda
            if recv_on_cpu and cpu_pg is not None:
                cpu_recv_flat = recv_buf.reshape(-1)
                if inplace:
                    cpu_send_tensor = cpu_recv_flat.narrow(
                        0, rank * data_num, data_num
                    ).contiguous()
                else:
                    cpu_send_tensor = send_buffers[i].reshape(-1).contiguous()
                gathered = [
                    torch.empty_like(cpu_send_tensor) for _ in range(world_size)
                ]
                torch.distributed.all_gather(gathered, cpu_send_tensor, group=cpu_pg)
                cpu_recv_flat.copy_(torch.cat(gathered, dim=0))
                continue
            gpu_recv = (
                recv_buf.to(torch.device("cuda", device_id))
                if recv_on_cpu
                else recv_buf
            )
            gpu_recv_flat = gpu_recv.reshape(-1)
            if inplace:
                send_tensor = gpu_recv_flat.narrow(
                    0, rank * data_num, data_num
                ).contiguous()
            else:
                send_t = send_buffers[i]
                send_tensor, _ = _ensure_cuda(send_t, device_id)
            torch.distributed.all_gather_into_tensor(
                gpu_recv_flat, send_tensor, group=pg
            )
            if recv_on_cpu:
                recv_buf.copy_(gpu_recv)

    _comm_ops_callbacks = (cpp_broadcast, cpp_allreduce, cpp_allgather)

    def synchronized_callback(callback_index: int):
        def wrapped(*args, **kwargs):
            with _comm_ops_lock:
                callbacks = _comm_ops_callbacks
                if callbacks is None:
                    raise RuntimeError("communication callbacks are not initialized")
                return callbacks[callback_index](*args, **kwargs)

        return wrapped

    librtp_compute_ops.register_comm_ops(
        synchronized_callback(0),
        synchronized_callback(1),
        synchronized_callback(2),
    )
    logging.info(
        f"Registered C++ comm ops callbacks (modes: {list(mode_to_group.keys())})"
    )


def _update_parallelism_config_for_topology(
    parallelism_config: ParallelismConfig, tp_size: int, dp_size: int
) -> None:
    """Update rank-derived topology fields in-place."""
    world_size = parallelism_config.world_size
    world_rank = parallelism_config.world_rank

    if tp_size <= 0 or dp_size <= 0:
        raise ValueError(
            f"tp_size and dp_size must be positive, got {tp_size}, {dp_size}"
        )
    if tp_size * dp_size != world_size:
        raise ValueError(
            f"tp_size * dp_size must equal world_size for hot topology switch: "
            f"{tp_size} * {dp_size} != {world_size}"
        )
    if (
        not parallelism_config.prefill_cp_config.is_enabled()
        and tp_size != parallelism_config.tp_size
    ):
        raise ValueError(
            "hot topology switch cannot change weight-sharded TP without "
            f"prefill CP enabled: current_tp={parallelism_config.tp_size}, "
            f"target_tp={tp_size}"
        )
    if parallelism_config.ffn_sp_size <= 0:
        raise ValueError(
            f"ffn_sp_size must be positive, got {parallelism_config.ffn_sp_size}"
        )
    if tp_size % parallelism_config.ffn_sp_size != 0:
        raise ValueError(
            f"tp_size must be divisible by ffn_sp_size: "
            f"{tp_size} % {parallelism_config.ffn_sp_size} != 0"
        )

    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = dp_size
    if parallelism_config.ep_size != 1:
        parallelism_config.ep_size = tp_size * dp_size
    parallelism_config.tp_rank = world_rank % tp_size
    parallelism_config.dp_rank = world_rank // tp_size
    parallelism_config.ep_rank = world_rank % parallelism_config.ep_size
    parallelism_config.ffn_tp_size = tp_size // parallelism_config.ffn_sp_size
    parallelism_config.ffn_tp_rank = (
        parallelism_config.tp_rank % parallelism_config.ffn_tp_size
    )
    parallelism_config.enable_sp = parallelism_config.ffn_sp_size > 1


def _update_parallelism_config_views_for_topology(tp_size: int, dp_size: int) -> None:
    """Update Python-side ParallelismConfig views that modules cached at init."""
    if _parallelism_config is None:
        return
    world_size = _parallelism_config.world_size
    world_rank = _parallelism_config.world_rank
    seen: set[int] = set()
    updated = 0
    updated_modules = 0

    def maybe_update(config: object) -> None:
        nonlocal updated
        if not isinstance(config, ParallelismConfig):
            return
        config_id = id(config)
        if config_id in seen:
            return
        seen.add(config_id)
        if config.world_size != world_size or config.world_rank != world_rank:
            return
        _update_parallelism_config_for_topology(config, tp_size, dp_size)
        updated += 1

    maybe_update(_parallelism_config)
    for module in list(_parallelism_config_modules):
        try:
            setattr(module, "parallelism_config", _parallelism_config)
            updated_modules += 1
        except Exception:
            continue

    logging.info(
        f"[rank: {world_rank}] updated {updated} Python ParallelismConfig views "
        f"and rebound {updated_modules} modules "
        f"for TP{tp_size}DP{dp_size}"
    )


def prewarm_parallel_topologies_from_env() -> None:
    spec = os.environ.get("RTP_LLM_PREWARM_TOPOLOGIES", "").strip()
    if not spec:
        return
    topologies = []
    for item in spec.split(","):
        tp, dp = (int(value) for value in item.lower().replace("x", ":").split(":"))
        if tp < 1 or dp < 1 or tp * dp != _parallelism_config.world_size:
            raise ValueError("resident topology must partition the existing world")
        topologies.append((tp, dp))
    if len(topologies) > 16:
        raise ValueError("at most 16 resident topologies are supported")
    for tp, dp in sorted(set(topologies)):
        prewarm_parallel_topology(tp, dp)


def prewarm_parallel_topology(tp_size: int, dp_size: int) -> None:
    """Create and warm a topology before model/KV allocation, on every rank."""
    global _group_map, _cpu_group_map
    if _parallelism_config is None or not torch.distributed.is_initialized():
        raise RuntimeError("distributed environment is not initialized")
    if not os.environ.get("RTP_LLM_PREWARM_TOPOLOGIES", "").strip():
        raise RuntimeError("resident topology prewarming must be configured at startup")
    if _parallelism_config.use_ub_comm or rocm_rccl.is_available_runtime():
        raise ValueError(
            "resident topology switching currently requires CUDA NCCL without user buffers"
        )
    if (
        tp_size < 1
        or dp_size < 1
        or tp_size * dp_size != _parallelism_config.world_size
    ):
        raise ValueError("resident topology must partition the existing world")
    active = (_parallelism_config.tp_size, _parallelism_config.dp_size)
    _resident_topologies.setdefault(active, (dict(_group_map), dict(_cpu_group_map)))
    target = (tp_size, dp_size)
    with _comm_ops_lock:
        try:
            if target not in _resident_topologies:
                _update_parallelism_config_for_topology(
                    _parallelism_config, tp_size, dp_size
                )
                _create_process_groups(
                    _parallelism_config, "nccl", timedelta(days=36500)
                )
                _resident_topologies[target] = (dict(_group_map), dict(_cpu_group_map))
            _group_map, _cpu_group_map = (
                dict(groups) for groups in _resident_topologies[target]
            )
            _update_parallelism_config_for_topology(
                _parallelism_config, tp_size, dp_size
            )
            # Exercise both model collectives and CPU control collectives before sizing KV memory.
            for mode in (Group.TP, Group.DP, Group.DP_AND_TP):
                group = _get_group(mode)
                value = torch.zeros(256, device="cuda")
                torch.distributed.all_reduce(value, group=group)
                outputs = [
                    torch.empty_like(value)
                    for _ in range(torch.distributed.get_world_size(group))
                ]
                torch.distributed.all_gather(outputs, value, group=group)
            for group in dict.fromkeys(_cpu_group_map.values()):
                torch.distributed.barrier(group=group)
            torch.cuda.synchronize()
            logging.info("Resident topology ready TP%dDP%d", tp_size, dp_size)
        finally:
            _group_map, _cpu_group_map = (
                dict(groups) for groups in _resident_topologies[active]
            )
            _update_parallelism_config_for_topology(_parallelism_config, *active)
            if active[0] > 1:
                init_symm_mem_communicator(_get_group(Group.TP))


def switch_parallel_topology(tp_size: int, dp_size: int) -> None:
    """Hot-switch active TP/DP process groups in an initialized worker.

    This is a stop-the-world primitive: callers must ensure all ranks in the
    world enter this function and no model work is in flight.
    """
    global _parallelism_config, _group_map, _cpu_group_map

    if _parallelism_config is None or not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed environment is not initialized; cannot switch topology"
        )

    world_rank = _parallelism_config.world_rank
    old_tp_size = _parallelism_config.tp_size
    old_dp_size = _parallelism_config.dp_size
    if (
        tp_size < 1
        or dp_size < 1
        or tp_size * dp_size != _parallelism_config.world_size
    ):
        raise ValueError("target topology must partition the existing world")
    if _resident_topologies and (tp_size, dp_size) not in _resident_topologies:
        raise ValueError("target topology was not warmed at startup")
    if old_tp_size == tp_size and old_dp_size == dp_size:
        logging.info(
            f"[rank: {world_rank}] switch_parallel_topology no-op: "
            f"tp_size={tp_size}, dp_size={dp_size}"
        )
        return

    switch_begin = time.monotonic()
    logging.info(
        f"[rank: {world_rank}] switching parallel topology "
        f"TP{old_tp_size}DP{old_dp_size} -> TP{tp_size}DP{dp_size}"
    )

    if os.environ.get("RTP_LLM_SWITCH_PRE_BARRIER", "0") == "1":
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        barrier_done = time.monotonic()
    else:
        barrier_done = switch_begin
    logging.info(f"[rank: {world_rank}] switch topology stage: pre-barrier done")
    if (
        os.environ.get("RTP_LLM_SWITCH_CUDA_SYNC", "0") == "1"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
    logging.info(f"[rank: {world_rank}] switch topology stage: cuda-sync done")
    # A C++ model thread can still be inside a Python communication callback
    # while the scheduler is finishing its drain phase.  Hold the same lock as
    # every registered callback so no process group is destroyed underneath an
    # in-flight broadcast/allreduce/allgather.
    with _comm_ops_lock:
        # ProcessGroup destruction is collective with ordering requirements.
        # The control-plane WORLD group is persistent, so use it to ensure no
        # rank starts destroying or recreating topology groups ahead of peers.
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
        if _parallelism_config.use_ub_comm:
            from rtp_llm.models_py.distributed.user_buffers import (
                destroy_user_buffers_communicator,
            )

            destroy_user_buffers_communicator()
        clear_symm_mem_communicator()
        logging.info(f"[rank: {world_rank}] switch topology stage: symm-mem cleared")
        if rocm_rccl.is_available_runtime():
            rocm_rccl.destroy_capture_comm()
        if not _resident_topologies:
            _destroy_active_topology_process_groups()
            torch.distributed.barrier(group=torch.distributed.group.WORLD)
        logging.info(
            f"[rank: {world_rank}] switch topology stage: old process-groups destroyed"
        )
        _update_parallelism_config_views_for_topology(tp_size, dp_size)
        logging.info(f"[rank: {world_rank}] switch topology stage: config updated")
        if _resident_topologies:
            _group_map, _cpu_group_map = (
                dict(groups) for groups in _resident_topologies[(tp_size, dp_size)]
            )
            if tp_size > 1:
                init_symm_mem_communicator(_get_group(Group.TP))
        else:
            _create_process_groups(_parallelism_config, "nccl", timedelta(days=36500))
        groups_done = time.monotonic()
        logging.info(
            f"[rank: {world_rank}] switch topology stage: process-groups ready"
        )
        _register_process_groups_to_cpp()
        register_done = time.monotonic()
        logging.info(
            f"[rank: {world_rank}] switch topology stage: cpp callbacks registered"
        )

    if rocm_rccl.is_available_runtime() and _parallelism_config.tp_size > 1:
        rocm_rccl.prepare_comm_if_needed(_parallelism_config, _get_group(Group.TP))
    rocm_done = time.monotonic()

    if _parallelism_config.use_ub_comm:
        init_user_buffers_environment(_parallelism_config)
    ub_done = time.monotonic()

    if os.environ.get("RTP_LLM_SWITCH_FINAL_BARRIER", "0") == "1":
        torch.distributed.barrier(group=torch.distributed.group.WORLD)
    switch_done = time.monotonic()
    logging.info(
        f"[rank: {world_rank}] switched parallel topology done: "
        f"{_parallelism_config.to_string()}, "
        f"barrier_ms={(barrier_done - switch_begin) * 1000:.2f}, "
        f"groups_ms={(groups_done - barrier_done) * 1000:.2f}, "
        f"register_ms={(register_done - groups_done) * 1000:.2f}, "
        f"rocm_ms={(rocm_done - register_done) * 1000:.2f}, "
        f"ub_ms={(ub_done - rocm_done) * 1000:.2f}, "
        f"final_barrier_ms={(switch_done - ub_done) * 1000:.2f}, "
        f"total_ms={(switch_done - switch_begin) * 1000:.2f}"
    )


def distributed_environment_initialized() -> bool:
    """Check if distributed environment is initialized.

    Returns:
        True if distributed environment is initialized, False otherwise
    """
    return torch.distributed.is_initialized()


def init_user_buffers_environment(parallelism_config: ParallelismConfig):
    """Initialize user buffers communicator for context parallelism."""
    from rtp_llm.models_py.utils.arch import is_cuda

    if parallelism_config.use_ub_comm and is_cuda():

        from rtp_llm.models_py.distributed.user_buffers import (
            init_user_buffers_communicator,
        )

        local_rank = parallelism_config.local_rank
        world_size = parallelism_config.world_size

        buffer_size = parallelism_config.prefill_cp_config.comm_buffer_size

        logging.info(
            f"[rank: {parallelism_config.world_rank}] Initializing user buffers communicator "
            f"with buffer_size: {buffer_size}, local_rank: {local_rank}, world_size: {world_size}"
        )
        init_user_buffers_communicator(
            _get_group(Group.TP), local_rank, world_size, buffer_size
        )


def destroy_distributed_environment():
    """Destroy distributed environment and clean up process groups.

    After calling this function, init_distributed_environment() can be called again
    to reinitialize the distributed environment.
    """
    global _group_map, _cpu_group_map, _parallelism_config, _initialized

    rank = torch.distributed.get_rank()
    logging.info(f"[rank: {rank}] Destroying distributed environment")

    from rtp_llm.models_py.utils.arch import is_cuda

    if is_cuda():
        from rtp_llm.models_py.distributed.user_buffers import (
            destroy_user_buffers_communicator,
        )

        destroy_user_buffers_communicator()

    try:
        import librtp_compute_ops

        if hasattr(librtp_compute_ops, "clear_comm_ops"):
            librtp_compute_ops.clear_comm_ops()
    except ImportError:
        pass

    # Clean up ROCm RCCL capture comm before destroying process groups,
    # so that re-init will bootstrap a fresh communicator instead of
    # reusing the stale one from the destroyed environment.
    if rocm_rccl.is_available_runtime():
        rocm_rccl.destroy_capture_comm()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _group_map.clear()
    _cpu_group_map.clear()
    _resident_groups.clear()
    _resident_topologies.clear()
    logging.info(f"[rank: {rank}] Distributed environment destroyed")
    _parallelism_config = None
    _initialized = False
    gc.collect()


def _get_group(group: Group) -> torch.distributed.ProcessGroup:
    """Get process group for the specified group type.

    This function checks if the distributed environment is initialized.
    If not initialized and _parallelism_config is available, it will attempt to initialize.

    Args:
        group: Group type (DP, TP, or DP_AND_TP)

    Returns:
        Process group for the specified group type

    Raises:
        RuntimeError: If distributed environment is not initialized and cannot be auto-initialized
        ValueError: If group type is invalid
    """
    global _parallelism_config, _initialized

    # Check if we need to initialize
    if not torch.distributed.is_initialized() or not _initialized:
        if _parallelism_config is not None:
            # Auto-initialize if we have the config
            logging.info(
                "Auto-initializing distributed environment from stored parallelism_config"
            )
            init_distributed_environment(_parallelism_config)
        else:
            raise RuntimeError(
                "Distributed environment is not initialized. "
                "Call init_distributed_environment(parallelism_config) first, "
                "or ensure parallelism_config is available for auto-initialization."
            )

    # Determine the actual key to use in _group_map
    group_key = group
    tp_size = _parallelism_config.tp_size
    dp_size = _parallelism_config.dp_size
    world_size = _parallelism_config.world_size
    if group == Group.DP and world_size != dp_size:
        tp_rank = torch.distributed.get_rank() % tp_size
        group_key = Group.DP.name + str(tp_rank)
    elif group == Group.TP and world_size != tp_size:
        dp_rank = torch.distributed.get_rank() // tp_size
        group_key = Group.TP.name + str(dp_rank)
    else:
        # DP_AND_TP always uses Group.DP_AND_TP as key
        group_key = Group.DP_AND_TP

    if group_key not in _group_map:
        raise ValueError(
            f"Process group {group_key} not found. Make sure init_distributed_environment() was called."
        )

    return _group_map[group_key]


# 需要注意：调用 send/recv 时如果某些 rank 没有操作，就没有对应的 ncclgroupstart/ncclgroupend
# 这样直接使用 torch 的 send/recv 是错误的。
def send(tensor: torch.Tensor, dst: int, group: Group) -> None:
    """Send a tensor to a destination rank.

    Args:
        tensor: Tensor to send
        dst: Destination global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.send(tensor, dst, group=process_group)


def recv(tensor: torch.Tensor, src: int, group: Group) -> torch.Tensor:
    """Receive a tensor from a source rank.

    Args:
        tensor: Tensor to receive into
        src: Source global rank
        group: Process group to use

    Returns:
        Received tensor (same as input tensor)
    """
    process_group = _get_group(group)
    torch.distributed.recv(tensor, src, group=process_group)
    return tensor


def broadcast(tensor: torch.Tensor, src: int, group: Group) -> None:
    """Broadcast a tensor from source rank to all ranks in the group.

    Args:
        tensor: Tensor to broadcast (will be modified on non-source ranks)
        src: Source global rank
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.broadcast(tensor, src, group=process_group)


def all_reduce(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """All-reduce a tensor across all ranks in the group.

    Args:
        tensor: Tensor to all-reduce (will be modified in-place)
        group: Process group to use

    Returns:
        All-reduced tensor (same as input tensor)
    """
    rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
    if rocm_rccl.should_use_capture_collectives(group == Group.TP):
        return rocm_rccl.capture_all_reduce(tensor, _get_group(group))

    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allreduce(
            tensor
        ):
            return symm_mem_comm.all_reduce(tensor)

    process_group = _get_group(group)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.SUM, group=process_group
    )
    return tensor


def all_gather(tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Gather tensors from all ranks in the group.

    Args:
        tensor: Tensor to gather from this rank
        group: Process group to use

    Returns:
        Concatenated tensor containing all gathered tensors
        (shape: [world_size * tensor.shape[0]] + list(tensor.shape)[1:])
    """
    rocm_rccl.ensure_capture_comm_ready(group == Group.TP)
    if rocm_rccl.should_use_capture_collectives(group == Group.TP):
        process_group = _get_group(group)
        return rocm_rccl.capture_all_gather(tensor, process_group)

    if group == Group.TP:
        symm_mem_comm = get_symm_mem_communicator()
        if symm_mem_comm is not None and symm_mem_comm.should_torch_symm_mem_allgather(
            tensor
        ):
            gathered = symm_mem_comm.all_gather(tensor)
            if gathered is not None:
                world_size = gathered.shape[0]
                return gathered.view(
                    [world_size * tensor.shape[0]] + list(tensor.shape)[1:]
                )

    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)

    tensor_list = torch.zeros(
        [world_size * tensor.shape[0]] + list(tensor.shape)[1:],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    torch.distributed.all_gather_into_tensor(tensor_list, tensor, group=process_group)
    return tensor_list

    # reference old implementation
    # tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # torch.distributed.all_gather(tensor_list, tensor, group=process_group)
    # return torch.cat(tensor_list, dim=0)


def reduce_scatter(input_tensor: torch.Tensor, group: Group) -> torch.Tensor:
    """Reduce-scatter a tensor across all ranks in the group.

    Reduces (sums) the input tensor across all ranks and scatters the result
    so that each rank receives a 1/world_size chunk of the reduced tensor.

    Args:
        input_tensor: Full-size tensor to reduce-scatter
            (shape: [world_size * chunk_size] + remaining_dims)
        group: Process group to use

    Returns:
        Scattered chunk of the reduced tensor for this rank
        (shape: [chunk_size] + remaining_dims)
    """
    process_group = _get_group(group)
    world_size = torch.distributed.get_world_size(process_group)
    assert input_tensor.shape[0] % world_size == 0, (
        f"reduce_scatter: input dim 0 ({input_tensor.shape[0]}) "
        f"must be divisible by world_size ({world_size})"
    )
    chunk_size = input_tensor.shape[0] // world_size
    output_tensor = torch.empty(
        [chunk_size] + list(input_tensor.shape[1:]),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.distributed.reduce_scatter_tensor(
        output_tensor,
        input_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    return output_tensor


def barrier(group: Group) -> None:
    """Barrier all ranks in the group.

    Args:
        group: Process group to use
    """
    process_group = _get_group(group)
    torch.distributed.barrier(group=process_group)


__all__ = [
    "Group",
    "init_distributed_environment",
    "init_user_buffers_environment",
    "switch_parallel_topology",
    "distributed_environment_initialized",
    "destroy_distributed_environment",
    "send",
    "recv",
    "broadcast",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "barrier",
]
