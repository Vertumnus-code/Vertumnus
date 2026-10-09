import json
import logging
import os
from typing import Any, Dict, List


def runtime_dp_addresses_file(server_port: int) -> str:
    if os.environ.get("PREFIX_TOPOLOGY_WRITER_URL"):
        directory = os.environ.get("PREFIX_TOPOLOGY_STATE_DIRECTORY", "")
        if not os.path.isabs(directory) or not os.path.isdir(directory):
            raise ValueError(
                "managed topology requires an existing absolute shared state directory"
            )
        return os.path.join(directory, f"rtp_llm_dp_addresses_{server_port}.json")
    return f"/tmp/rtp_llm_dp_addresses_{server_port}.json"


def compute_dp_addresses(
    all_rank_addresses: List[str], tp_size: int, dp_size: int
) -> List[str]:
    if tp_size <= 0 or dp_size <= 0:
        raise ValueError(f"tp_size and dp_size must be positive: {tp_size}, {dp_size}")
    addresses = [
        address
        for rank, address in enumerate(all_rank_addresses)
        if rank % tp_size == 0 and rank // tp_size < dp_size
    ]
    if len(addresses) != dp_size:
        raise ValueError(
            f"cannot derive {dp_size} DP addresses from {len(all_rank_addresses)} "
            f"rank addresses with tp_size={tp_size}: {addresses}"
        )
    return addresses


def write_runtime_dp_addresses(
    path: str, dp_addresses: List[str], tp_size: int, dp_size: int
) -> None:
    payload = {
        "dp_addresses": dp_addresses,
        "tp_size": tp_size,
        "dp_size": dp_size,
    }
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
    logging.info(
        "updated runtime DP addresses file %s: tp_size=%s dp_size=%s addresses=%s",
        path,
        tp_size,
        dp_size,
        dp_addresses,
    )


def read_runtime_dp_addresses(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
