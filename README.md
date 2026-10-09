# Vertumnus

Vertumnus is an LLM serving system built on RTP-LLM that adapts context parallelism (CP) to changing workloads. It coordinates request scheduling, worker reconfiguration, and prefix cache management to balance prefill latency and GPU utilization.

- **CP-aware scheduling.** The `CP_AWARE_TTFT` strategy selects workers using predicted queueing delay, cache-aware prefill latency, and the additional GPU time required by a larger CP degree.
- **Resident worker merge and split.** Workers switch between adjacent CP degrees, such as one CP4 worker and two CP2 workers, while keeping model weights resident. Prewarmed communication groups and resident cache preservation support fast reconfiguration.
- **Prefix cache management.** A global logical prefix trie tracks demand and confirmed cache replicas. The system coordinates replication across and within CP degrees, reclaims cold prefixes, and keeps cache state consistent with worker topology changes.

FlexLB implements the scheduling and control plane; the RTP-LLM engine executes inference, cache operations, and native worker reconfiguration.

## Quick start

The example below uses Linux x86_64, Python 3.10, CUDA 12.9, and a local Qwen3 MoE model. Use a GPU with enough memory for the model and KV cache. Build and install the engine from this repository in a compatible build environment; adjust the Python and CUDA paths in [`.bazelrc`](.bazelrc) as needed.

```bash
# Run from the repository root.
bazelisk build //rtp_llm:rtp_llm --config=cuda12_9
python -m pip install bazel-bin/rtp_llm/rtp_llm-*-cp310-cp310-manylinux1_x86_64.whl

# Run outside the source tree to use the installed package.
cd ..
MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
CUDA_VISIBLE_DEVICES=0 python -m rtp_llm.start_server \
  --checkpoint_path "$MODEL_PATH" \
  --tokenizer_path "$MODEL_PATH" \
  --model_type qwen_3_moe \
  --tp_size 1 --dp_size 1 --world_size 1 \
  --max_seq_len 8192 --frontend_server_count 1 \
  --start_port 8088
```

Once the service is ready, send a request from another terminal:

```bash
curl --fail http://localhost:8088/health
curl --fail http://localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen_3_moe",
    "messages": [{"role": "user", "content": "Explain context parallelism briefly."}],
    "max_tokens": 128,
    "stream": false
  }'
```

For CP-aware cluster serving, build FlexLB with JDK 21 and launch it with your worker and control-plane configuration:

```bash
# Run from the repository root.
cd rtp_llm/flexlb
mvn -Popensource clean package
export FLEXLB_CONFIG='{"loadBalanceStrategy":"CP_AWARE_TTFT","enableQueueing":true}'
export FLEXLB_CP_COST_MODEL=GPU_SECONDS
export FLEXLB_CP_ADDITIONAL_GPU_SECONDS=true
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
  --spring.config.additional-location=file:/path/to/cluster.yml
```

The single-worker example starts the inference engine. Dynamic merge/split and adaptive prefix management additionally require registered workers, ZooKeeper, topology settings, and cache-transfer endpoints. See [worker reconfiguration](docs/backend/worker_split_merge.md) and [prefix cache settings](rtp_llm/flexlb/flexlb-cache/src/main/java/org/flexlb/cache/prefix/PrefixCacheSettings.java) for those options.

## License

This project builds on RTP-LLM and is licensed under [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and the accompanying third-party license files for upstream attribution.
