#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPromptConstructor.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "autil/TimeUtility.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
#include <pybind11/embed.h>
#include <algorithm>
#include <memory>
#include <thread>
#include <random>

#if USING_CUDA
#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAStream.h"
#endif

#ifdef __linux__
#include <malloc.h>
#endif

using namespace std;
namespace rtp_llm {

namespace {
// 释放glibc缓存的host内存，将其归还给操作系统
// 在模型加载完成后调用，可以显著减少常驻内存占用
void releaseHostMemoryCache() {
#ifdef __linux__
    // malloc_trim(0) 会释放所有可以释放的内存回操作系统
    // 这对于checkpoint加载后释放临时分配的大量CPU内存很重要
    int result = malloc_trim(0);
    RTP_LLM_LOG_INFO("Released host memory cache to OS (malloc_trim returned %d)", result);
#else
    RTP_LLM_LOG_DEBUG("malloc_trim not available on this platform");
#endif
}

ParallelismConfig makeSwitchedParallelismConfig(ParallelismConfig config, int64_t tp_size, int64_t dp_size) {
    RTP_LLM_CHECK_WITH_INFO(tp_size > 0 && dp_size > 0, "tp_size and dp_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(tp_size * dp_size == config.world_size,
                            "tp_size * dp_size must equal world_size, got tp=%ld dp=%ld world=%ld",
                            tp_size,
                            dp_size,
                            config.world_size);
    RTP_LLM_CHECK_WITH_INFO(config.prefill_cp_config.is_enabled() || tp_size == config.tp_size,
                            "hot topology switch cannot change weight-sharded TP without prefill CP enabled; "
                            "got current_tp=%ld target_tp=%ld",
                            config.tp_size,
                            tp_size);
    RTP_LLM_CHECK_WITH_INFO(config.ffn_sp_size > 0, "ffn_sp_size must be positive");
    RTP_LLM_CHECK_WITH_INFO(tp_size % config.ffn_sp_size == 0,
                            "tp_size must be divisible by ffn_sp_size, got tp=%ld ffn_sp=%ld",
                            tp_size,
                            config.ffn_sp_size);

    config.tp_size = tp_size;
    config.dp_size = dp_size;
    if (config.ep_size != 1) {
        config.ep_size = tp_size * dp_size;
    }
    config.tp_rank     = config.world_rank % config.tp_size;
    config.dp_rank     = config.world_rank / config.tp_size;
    config.ep_rank     = config.world_rank % config.ep_size;
    config.ffn_tp_size = config.tp_size / config.ffn_sp_size;
    config.ffn_tp_rank = config.tp_rank % config.ffn_tp_size;
    config.enable_sp   = config.ffn_sp_size > 1;
    return config;
}
}  // anonymous namespace

NormalEngine::NormalEngine(const EngineInitParams&                       params,
                           std::unique_ptr<ProposeModelEngineInitParams> propose_params):
    EngineBase(params),
    model_config_(params.model_config_),
    parallelism_config(params.parallelism_config),
    runtime_config(params.runtime_config),
    eplb_config(params.eplb_config),
    pd_sep_config(params.pd_sep_config),
    profiling_debug_logging_config(params.profiling_debug_logging_config),
    kv_cache_config(params.kv_cache_config),
    ffn_disaggregate_config(params.ffn_disaggregate_config),
    model_specific_config(params.model_specific_config),
    sp_config(params.sp_config),
    metrics_reporter_(params.metrics_reporter),
    propose_params_(std::move(propose_params)),
    step_profiler_(params.profiling_debug_logging_config.torch_cuda_profiler_dir,
                   params.parallelism_config.dp_rank * params.parallelism_config.tp_size
                       + params.parallelism_config.tp_rank) {
    RTP_LLM_LOG_INFO(__PRETTY_FUNCTION__);
#if !USING_CUDA
    // On ROCm, this constructor runs on a gRPC handler thread that defaults to
    // GPU 0. Set the correct device so all GPU allocations (KV cache, etc.) go
    // to the right device.  The guard is scoped to the constructor body.
    c10::DeviceGuard ctor_device_guard(
        c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(parallelism_config.local_rank)));
    RTP_LLM_LOG_INFO("ROCm NormalEngine ctor: set device to %d", parallelism_config.local_rank);
#endif

    std::optional<WarmUpResult> warm_up_result = std::nullopt;
#if USING_CUDA
    if (runtime_config.warm_up && (!model_config_.mm_model_config.is_multimodal)
        && !ffn_disaggregate_config.enable_ffn_disaggregate) {
        // warm up
        RTP_LLM_LOG_INFO("warm up (max_context_batch_size %d, max_seq_len %d calculate_loss %d) query begin",
                         runtime_config.fifo_scheduler_config.max_context_batch_size,
                         model_config_.max_seq_len,
                         int(runtime_config.warm_up_with_loss));
        warm_up_result = warmUp(params);
        RTP_LLM_LOG_INFO(
            "warm up done, max runtime used memory: %ld bytes (%ld MiB), device reserved memory: %ld bytes (%ld MiB)",
            warm_up_result->max_used_memory,
            warm_up_result->max_used_memory / 1024 / 1024,
            warm_up_result->device_reserved_bytes,
            warm_up_result->device_reserved_bytes / 1024 / 1024);
    } else {
        RTP_LLM_LOG_INFO("skip warm up.");
    }
#else
    RTP_LLM_LOG_INFO("skip warm up on non-CUDA platform.");
#endif
    initCacheManager(warm_up_result, params.cache_store_config);
    RTP_LLM_LOG_INFO("create cache manager done");

    initExecutor(params, propose_params_);
    if (propose_params_) {
        reserve_step_ = propose_params_->gen_num_per_circle + 1;
    } else {
        reserve_step_ = 0;
    }

    RTP_LLM_LOG_INFO("create normal executor done");

    // 释放模型加载过程中使用的临时host内存
    // 此时checkpoint已加载完成，可以将glibc缓存的内存归还给操作系统
    releaseHostMemoryCache();

    initScheduler();
    (void)startLoop();
}

void NormalEngine::initExecutor(const EngineInitParams&                        params,
                                std::unique_ptr<ProposeModelEngineInitParams>& propose_params) {
    if (propose_params_) {
        executor_.reset(new MtpExecutor(params,
                                        propose_params,
                                        resource_context_.cache_manager,
                                        mla_ops_type_,
                                        kv_cache_group_num_,
                                        kv_cache_layer_to_group_));
    } else {
        executor_.reset(new NormalExecutor(params,
                                           resource_context_.cache_manager,
                                           false,
                                           false,
                                           0,
                                           mla_ops_type_,
                                           kv_cache_group_num_,
                                           kv_cache_layer_to_group_));
    }
}

void NormalEngine::initScheduler() {
    if (runtime_config.use_batch_decode_scheduler) {
        scheduler_.reset(new BatchDecodeScheduler(
            runtime_config, resource_context_.cache_manager, metrics_reporter_, parallelism_config.dp_rank));
        RTP_LLM_LOG_INFO("create batch decode scheduler done");
    } else {
        scheduler_.reset(new FIFOScheduler(runtime_config,
                                           model_config_,
                                           pd_sep_config,
                                           parallelism_config,
                                           model_specific_config,
                                           resource_context_.cache_manager,
                                           metrics_reporter_));
        RTP_LLM_LOG_INFO("create fifo scheduler done");
    }
}

NormalEngine::~NormalEngine() {
    RTP_LLM_LOG_INFO("destory normal engine");
    (void)stop();
}

absl::StatusOr<GenerateStreamPtr> NormalEngine::preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                       preRunMode                            mode) {
    auto stream = std::make_shared<NormalGenerateStream>(generate_input,
                                                         model_config_,
                                                         runtime_config,
                                                         resource_context_,
                                                         nullptr,
                                                         0,
                                                         mode == preRunMode::prefill_warm_up);
    if (mode == preRunMode::decode_warm_up) {
        stream->setIsContextStream(false);
        size_t seq_size_per_block = model_config_.attn_config.tokens_per_block;
        size_t reserved_blocks    = (stream->seqLength() + seq_size_per_block - 1) / seq_size_per_block + reserve_step_;
        stream->fakeInitKVBlock(reserved_blocks);
    } else if (mode == preRunMode::build_system_prompt) {
        stream->setReserveStep(reserve_step_);
        THROW_IF_STATUS_ERROR(stream->initKVBlock());
    };
    std::list<GenerateStreamPtr> streams{stream};
    THROW_IF_STATUS_ERROR(executor_->process(streams));
    return stream;
}

int64_t NormalEngine::getLastScheduleTime() {
    return scheduler_->lastScheduleTime();
}

WarmUpResult NormalEngine::warmUp(const EngineInitParams& params) {
    if (runtime_config.use_batch_decode_scheduler) {
        if (runtime_config.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type == 0) {
            return decodeWarmUp(params);
        } else {
            return prefillWarmUp(params);
        }
    }
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::PREFILL) {
        return prefillWarmUp(params);
    } else if (pd_sep_config.role_type == RoleType::DECODE) {
        return decodeWarmUp(params);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
        return {};
    }
}

std::shared_ptr<GenerateInput> NormalEngine::makeFakeInput(size_t seq_len) {
    std::shared_ptr<GenerateInput> fake_input = make_shared<GenerateInput>();
    fake_input->generate_config               = make_shared<GenerateConfig>();
    size_t token_size                         = model_config_.embedding_size ?
                                                    std::min(model_config_.embedding_size, model_config_.vocab_size) :
                                                    model_config_.vocab_size;
    fake_input->input_ids              = torch::randint(0, (int64_t)token_size, {(int64_t)seq_len}, torch::kInt32);
    fake_input->begin_time_us          = autil::TimeUtility::currentTimeInMicroSeconds();
    fake_input->generate_config->top_k = 1;

    return fake_input;
}

WarmUpResult NormalEngine::prefillWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("prefillWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.fifo_scheduler_config.max_context_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);
    executor_.reset(new NormalExecutor(
        params, nullptr, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::prefill_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

WarmUpResult NormalEngine::decodeWarmUp(const EngineInitParams& params) {
#if !USING_CUDA
    RTP_LLM_FAIL("decodeWarmUp is not supported on non-CUDA platforms");
    return {};
#else
    auto fake_input                                   = makeFakeInput((size_t)model_config_.max_seq_len - 1);
    fake_input->generate_config->num_return_sequences = runtime_config.max_generate_batch_size;
    fake_input->generate_config->calculate_loss       = int(runtime_config.warm_up_with_loss);
    rtp_llm::setTraceMemory(true);

    auto cache_config               = CacheConfigCreator::createBasicConfig(model_config_, parallelism_config);
    cache_config.seq_size_per_block = model_config_.attn_config.tokens_per_block;
    cache_config.block_num          = 5;
    ParallelismConfig temp_parallelism_config;
    RuntimeConfig     temp_runtime_config;
    auto              cache_manager = make_shared<KVCacheManager>(
        cache_config, true, nullptr, KVCacheConfig{}, temp_parallelism_config, temp_runtime_config);
    if (!cache_manager->init()) {
        RTP_LLM_FAIL("init kv cache manager failed in decodeWarmUp");
    }
    executor_.reset(new NormalExecutor(
        params, cache_manager, true, false, 0, mla_ops_type_, kv_cache_group_num_, kv_cache_layer_to_group_));
    THROW_IF_STATUSOR_ERROR(preRun(fake_input, preRunMode::decode_warm_up));
    const auto max_consumed = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    rtp_llm::setTraceMemory(false);
    (void)executor_.reset(nullptr);
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto device_status = getGpuExecStatus();
    return WarmUpResult({device_status.device_memory_status.available_bytes, max_consumed});
#endif
}

std::shared_ptr<GenerateStream> NormalEngine::createMinFakeStream(int32_t max_new_tokens) {
    RTP_LLM_LOG_DEBUG("create min fake query");
    auto fake_input                             = makeFakeInput(1);
    fake_input->generate_config->max_new_tokens = max_new_tokens;
    fake_input->fake_query                      = true;
    auto stream                                 = makeStream(fake_input);
    stream->setIsFakeStream(true);
    stream->setMetricsReporter(nullptr);
    stream->fakeInitKVBlock();
    if (pd_sep_config.role_type == RoleType::PDFUSION || pd_sep_config.role_type == RoleType::DECODE) {
        auto new_tokens = torch::zeros({1, 1}, torch::kInt32);

        StreamUpdateInfo update_info{new_tokens,
                                     1,
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     torch::Tensor(),
                                     false};
        stream->update(update_info);
    }
    return stream;
}

void NormalEngine::initCacheManager(std::optional<WarmUpResult> warm_up_result,
                                    const CacheStoreConfig&     cache_store_config) {
    if (propose_params_ && propose_params_->draftModel()) {
        auto config = CacheConfigCreator::createSpConfig(model_config_,
                                                         propose_params_->getEngineInitParams().model_config_,
                                                         parallelism_config,
                                                         runtime_config,
                                                         kv_cache_config,
                                                         sp_config,
                                                         warm_up_result,
                                                         isMTPEagle(),
                                                         isEagle());

        resource_context_.cache_manager = make_shared<KVCacheManager>(config,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      kv_cache_config,
                                                                      parallelism_config,
                                                                      runtime_config,
                                                                      sp_config,
                                                                      pd_sep_config,
                                                                      cache_store_config);
        resource_context_.role_type     = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }

        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
    } else {
        auto result = CacheConfigCreator::createConfig(
            model_config_, parallelism_config, runtime_config, kv_cache_config, warm_up_result);
        RTP_LLM_LOG_INFO("create cache manager with config %s", result.debugString().c_str());
        RTP_LLM_LOG_INFO("create cache manager with block nums %d, block size %ld KB",
                         result.block_num,
                         result.block_size_bytes / 1024);
        RTP_LLM_LOG_INFO("create cache manager with linear step %d", result.linear_step);
        resource_context_.cache_manager = make_shared<KVCacheManager>(result,
                                                                      false,
                                                                      metrics_reporter_,
                                                                      kv_cache_config,
                                                                      parallelism_config,
                                                                      runtime_config,
                                                                      sp_config,
                                                                      pd_sep_config,
                                                                      cache_store_config);
        resource_context_.role_type     = pd_sep_config.role_type;
        if (!resource_context_.cache_manager->init()) {
            RTP_LLM_FAIL("init kv cache manager failed");
        }
        const auto& cache_cfg    = resource_context_.cache_manager->cacheConfig();
        kv_cache_group_num_      = cache_cfg.groupNums();
        kv_cache_layer_to_group_ = cache_cfg.layer_to_group_id;
    }
}

absl::Status NormalEngine::initSystemPrompt() {
    if (!kv_cache_config.multi_task_prompt_tokens.empty()) {
        resource_context_.reuse_cache = true;
        CHECK_AND_RETURN_REF(
            system_prompt_param,
            SystemPromptConstructor::construct(
                kv_cache_config, this, resource_context_.cache_manager.get(), parallelism_config.tp_rank == 0));
        resource_context_.system_prompt.reset(new SystemPrompt(system_prompt_param));
    }

    return absl::OkStatus();
}

KVCacheInfo NormalEngine::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    return resource_context_.cache_manager->getKVCacheInfo(latest_version, need_cache_keys);
}

absl::Status NormalEngine::startLoop() {
    // A follower can become a DP leader after a split without restarting.
    resource_context_.initCacheConfig(kv_cache_config, runtime_config.fifo_scheduler_config, model_config_.max_seq_len);
    if (parallelism_config.tp_rank == 0) {
        RTP_LLM_LOG_INFO("start init system prompt");
        THROW_IF_STATUS_ERROR(initSystemPrompt());
        RTP_LLM_LOG_INFO("init system prompt done");
    }
    RTP_LLM_LOG_INFO("start normal engine loop");
    running_     = true;
    loop_thread_ = autil::Thread::createThread(std::bind(&NormalEngine::loop, this), "normal_engine_loop");
    return absl::OkStatus();
}

absl::Status NormalEngine::stop() {
    RTP_LLM_LOG_INFO("stop normal engine");
    running_ = false;
    RETURN_IF_STATUS_ERROR(scheduler_->stop());
    loop_thread_->join();
    return absl::OkStatus();
}

void NormalEngine::loop() {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_INFO("loop begin");
    cudaPreRun(getDeviceId());
    while (running_) {
        auto status = step();
        if (!status.ok()) {
            RTP_LLM_LOG_ERROR("step running error: %s", status.ToString().c_str());
            THROW_IF_STATUS_ERROR(trySaveStepError());
        }
    }
}

absl::Status NormalEngine::trySaveStepError() const {
    return absl::UnimplementedError("can not save yet!");
}

std::shared_ptr<GenerateStream> NormalEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    return stream;
}

void NormalEngine::enqueue(std::shared_ptr<GenerateStream>& stream) {
    stream->setReserveStep(reserve_step_);
    touchPrefixAtAdmission(stream);
    (void)scheduler_->enqueue(stream);
}

void NormalEngine::touchPrefixAtAdmission(const std::shared_ptr<GenerateStream>& stream) {
    if (stream && !stream->hasError() && stream->isContextStream() && stream->reuseCache()
        && stream->enableDeviceCache() && resource_context_.cache_manager) {
        resource_context_.cache_manager->touchPrefixAtAdmission(stream->completeTokenIdsPtr());
    }
}

std::shared_ptr<GenerateStream> NormalEngine::enqueue(const std::shared_ptr<GenerateInput>& input) {
    std::shared_ptr<GenerateStream> stream = std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
    stream->setReserveStep(reserve_step_);
    touchPrefixAtAdmission(stream);
    (void)scheduler_->enqueue(stream);
    return stream;
}

std::vector<std::shared_ptr<GenerateStream>>
NormalEngine::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    std::vector<std::shared_ptr<GenerateStream>> streams;
    streams.reserve(inputs.size());
    for (auto& inp : inputs) {
        auto stream = std::make_shared<NormalGenerateStream>(
            inp, model_config_, runtime_config, resource_context_, metrics_reporter_);
        stream->setReserveStep(reserve_step_);
        touchPrefixAtAdmission(stream);
        streams.push_back(stream);
    }
    return scheduler_->batchEnqueue(streams);
}

absl::Status NormalEngine::step() {
    RTP_LLM_PROFILE_SCOPE("engine.normal.step_work");
    auto wait_if_switch_paused = [this]() {
        while (pause_.load() && running_) {
            // wait 50ms if system paused.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    };
    wait_if_switch_paused();
    if (!running_) {
        return absl::OkStatus();
    }

    list<GenerateStreamPtr> streams;
    if (parallelism_config.tp_rank == 0 && !ffn_disaggregate_config.is_ffn_service()) {
        const bool prefix_install_pending = resource_context_.cache_manager->prefixRankInstallPending();
        const bool prefix_control_step    = prefix_install_pending && !last_prefix_control_step_
                                         && parallelism_config.dp_size == 1 && parallelism_config.ep_size == 1
                                         && resource_context_.cache_manager->prefixInventoryEnabled()
                                         && resource_context_.cache_manager->supportsPrefixRankReplication();
        last_prefix_control_step_ = prefix_control_step;
        if (prefix_install_pending) {
            wakeupForPrefixCacheInstall();
        }
        if (!prefix_control_step) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.schedule(reserve_step=%d)", reserve_step_);
            CHECK_AND_ASSIGN(streams, scheduler_->schedule());
        }
        if (parallelism_config.dp_size > 1) {
            RTP_LLM_PROFILE_SCOPE("engine.normal.may_add_fake_stream_work");
            mayAddFakeStream(streams);
        }
        // When TP > 1, all ranks must enter process() together so that
        // tpSyncModelInputs (collective broadcast) does not deadlock.
        // The skip_run flag inside process() handles the "no work" case.
        if (streams.empty() && parallelism_config.tp_size <= 1 && !prefix_control_step) {
            if (resource_context_.cache_manager->prefixInventoryEnabled()) {
                std::lock_guard<std::mutex> lock(execution_mutex_);
                if (!pause_.load()) {
                    auto boundary = resource_context_.cache_manager->prefixInventoryBoundary();
#if USING_CUDA
                    c10::cuda::getCurrentCUDAStream().synchronize();
#endif
                    resource_context_.cache_manager->confirmPrefixInventory(boundary, true);
                }
            }
            return absl::OkStatus();
        }
    }

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int64_t      step_begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
    absl::Status status             = absl::OkStatus();

    // If any stream in this batch requested gen_timeline AND no profiling session is
    // already active, configure + tick BEFORE process() so the profiler is up before
    // the actual work runs. This guarantees the trace captures THIS step's work, not
    // the next step's. If a session is already active (e.g. external StartProfile RPC),
    // skip — first-come-first-served.
    if (!step_profiler_.enabled()) {
        for (const auto& stream : streams) {
            if (stream && stream->genTimeline()) {
                const auto& cfg = stream->generateConfig();
                step_profiler_.configure(true, cfg->profile_trace_name, 0, cfg->profile_step);
                step_profiler_.tick();  // start profiler now (start_step=0)
                break;
            }
        }
    }

    while (true) {
        wait_if_switch_paused();
        if (!running_) {
            return absl::OkStatus();
        }
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.normal.execute(stream_size=%zu)", streams.size());
        std::unique_lock<std::mutex> lock(execution_mutex_);
        if (pause_.load()) {
            lock.unlock();
            continue;
        }
        const auto pause_at_ms = topology_switch_pause_at_ms_.load();
        const bool request_topology_switch_pause =
            topology_switch_armed_.load()
            && (pause_at_ms <= 0 || autil::TimeUtility::currentTimeInMilliSeconds() >= pause_at_ms);
        if (executor_) {
            executor_->setTopologySwitchPauseRequested(request_topology_switch_pause);
        }
        std::vector<PrefixReuseEvent>                                        prefix_events;
        std::vector<std::pair<std::unique_ptr<PrefixCopyLedger::Use>, bool>> prefix_copy_uses;
        for (const auto& stream : streams) {
            if (stream && resource_context_.cache_manager->prefixInventoryEnabled()
                && (!stream->adapterName().empty() || stream->generateInput()->multimodal_inputs.has_value()
                    || stream->generateInput()->multimodal_features.has_value()
                    || stream->generateInput()->prefix_length != 0)) {
                resource_context_.cache_manager->fencePrefixInventory();
            }
            if (stream && stream->isContextStream() && stream->iterCount() == 0
                && pd_sep_config.role_type != RoleType::DECODE) {
                auto use = resource_context_.cache_manager->capturePrefixCopyUse(
                    stream->kvCache(),
                    stream->localReuseLength(),
                    stream->streamId(),
                    stream->generateInput()->prefix_cache_attempt);
                if (use) {
                    prefix_copy_uses.emplace_back(std::move(use), stream->memoryReuseLength() == 0);
                }
            }
            if (stream && stream->isContextStream() && stream->iterCount() == 0
                && pd_sep_config.role_type != RoleType::DECODE
                && !stream->generateInput()->prefix_cache_attempt.empty()) {
                prefix_events.push_back({0,
                                         stream->streamId(),
                                         std::max(0, stream->localReuseLength() - stream->memoryReuseLength()),
                                         stream->inputLength(),
                                         true,
                                         stream->generateInput()->prefix_cache_attempt});
            }
        }
        status                      = executor_->process(streams);
        bool prefix_copy_step_ready = status.ok();
        if (executor_->prefixInventoryStepEnabled()) {
            RTP_LLM_PROFILE_SCOPE("prefix.inventory_step");
            auto boundary = resource_context_.cache_manager->prefixInventoryBoundary();
#if USING_CUDA
            {
                RTP_LLM_PROFILE_SCOPE("prefix.inventory_stream_sync");
                c10::cuda::getCurrentCUDAStream().synchronize();
            }
#endif
            bool ready = status.ok();
            if (parallelism_config.tp_size > 1) {
                RTP_LLM_PROFILE_SCOPE("prefix.inventory_all_rank_ready");
                auto readiness = torch::full({1}, ready ? 1 : 0, torch::TensorOptions().dtype(torch::kInt32));
                readiness      = execAllReduce({readiness, ReduceOp::Sum, false, ParallelMode::TP}).buffer;
                ready          = readiness.item<int>() == parallelism_config.tp_size;
            }
            resource_context_.cache_manager->confirmPrefixInventory(boundary, ready);
            prefix_copy_step_ready = ready;
        }
        for (auto& use : prefix_copy_uses) {
            use.first->finish(prefix_copy_step_ready && use.second);
        }
        if (status.ok()) {
            for (const auto& event : prefix_events) {
                resource_context_.cache_manager->recordPrefixReuse(
                    event.request_id, event.device_reuse_tokens, event.input_tokens, event.attempt_id);
            }
        }
        break;
    }
    topology_switch_step_count_.fetch_add(1);
    if (executor_ && executor_->lastTopologySwitchPauseRequested()) {
        RTP_LLM_LOG_INFO("parallel topology switch observed from synced executor input, pause now");
        pause_ = true;
        if (!status.ok()) {
            resumeParallelTopologySwitch();
        }
    }

    // tick profiler after process() to count this step (and stop when num_steps reached).
    // All TP ranks synchronize inside process() via NCCL, so stop happens at aligned points.
    step_profiler_.tick();

    // report step metrics
    if (parallelism_config.tp_rank == 0) {
        RTP_LLM_PROFILE_SCOPE("engine.normal.report_metrics_work");
        auto step_latency = autil::TimeUtility::currentTimeInMicroSeconds() - step_begin_time_us;
        reportMetrics({step_latency});
    }

    return status;
}

bool NormalEngine::updateEplbConfig(const EPLBConfig& config) {
    if (executor_) {
        return executor_->updateEplbConfig(config);
    }
    return true;
}

absl::Status NormalEngine::restart() {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (!topology_switch_id_.empty() || topology_switch_armed_.load()
        || prefix_topology_phase_.load() != PrefixTopologyPhase::NONE) {
        return absl::FailedPreconditionError("cannot restart outside the active topology protocol");
    }
    return EngineBase::restart();
}

absl::Status NormalEngine::claimTopologySwitchLocked(const std::string& switch_id) {
    if ((!topology_switch_id_.empty() && topology_switch_id_ != switch_id)
        || (topology_switch_id_.empty() && topology_switch_identities_.count(switch_id) != 0)) {
        return absl::FailedPreconditionError("topology switch identity is active or retired");
    }
    if (!switch_id.empty() && topology_switch_id_.empty()) {
        if (topology_switch_identities_.size() >= MAX_TOPOLOGY_SWITCH_IDENTITIES) {
            return absl::ResourceExhaustedError("topology switch identity history is full; replace the worker");
        }
        topology_switch_identities_.insert(switch_id);
    }
    topology_switch_id_ = switch_id;
    return absl::OkStatus();
}

absl::Status NormalEngine::drainParallelTopologySwitch(int64_t            timeout_ms,
                                                       const std::string& switch_id,
                                                       const std::string& expected_epoch) {
    {
        std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
        if (switch_id.empty() || switch_id.size() > 128) {
            return absl::InvalidArgumentError("topology drain identity is required");
        }
        if (!expected_epoch.empty()
            && expected_epoch != resource_context_.cache_manager->getKVCacheInfo(-1, false).inventory.epoch) {
            return absl::FailedPreconditionError("topology drain epoch changed after participant discovery");
        }
        if (prefix_topology_phase_.load() != PrefixTopologyPhase::NONE || topology_switch_armed_.load()) {
            return absl::FailedPreconditionError("topology drain conflicts with active or retired switch");
        }
        const auto identity_status = claimTopologySwitchLocked(switch_id);
        if (!identity_status.ok()) {
            return identity_status;
        }
        scheduler_->setDrainMode(true);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 300000);
    while (true) {
        {
            std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
            if (topology_switch_id_ != switch_id) {
                return absl::CancelledError("topology scheduler drain cancelled");
            }
            if (scheduler_->onflightStreams() == 0) {
                return absl::OkStatus();
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return absl::DeadlineExceededError("timeout draining requests on a topology rank");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

absl::Status NormalEngine::releaseParallelTopologySwitch(const std::string& switch_id) {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (switch_id.empty() || switch_id != last_topology_switch_id_ || !topology_switch_id_.empty()
        || prefix_topology_phase_.load() != PrefixTopologyPhase::NONE || pause_.load()) {
        return absl::FailedPreconditionError("cannot release routing before the matching topology switch settles");
    }
    scheduler_->setDrainMode(false);
    return absl::OkStatus();
}

absl::Status NormalEngine::armParallelTopologySwitch(int64_t pause_at_ms, const std::string& switch_id) {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (pause_at_ms < 0) {
        return absl::InvalidArgumentError("pause_at_ms must be non-negative");
    }
    if (switch_id.size() > 128 || (resource_context_.cache_manager->prefixInventoryEnabled() && switch_id.empty())) {
        return absl::InvalidArgumentError("prefix topology switch requires a bounded switch identity");
    }
    if (prefix_topology_phase_.load() == PrefixTopologyPhase::QUARANTINED) {
        return absl::FailedPreconditionError("topology switch is active, retired or quarantined");
    }
    const auto identity_status = claimTopologySwitchLocked(switch_id);
    if (!identity_status.ok()) {
        return identity_status;
    }
    topology_switch_pause_at_ms_.store(pause_at_ms);
    topology_switch_armed_.store(true);
    RTP_LLM_LOG_INFO("parallel topology switch armed, pause_at_ms=%ld now_ms=%ld",
                     pause_at_ms,
                     autil::TimeUtility::currentTimeInMilliSeconds());
    return absl::OkStatus();
}

absl::Status NormalEngine::prepareParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id) {
    const auto effective_timeout = timeout_ms > 0 ? timeout_ms : 300000;
    const auto deadline          = std::chrono::steady_clock::now() + std::chrono::milliseconds(effective_timeout);
    while (true) {
        {
            std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
            if (switch_id != topology_switch_id_) {
                return absl::FailedPreconditionError("prepare topology switch identity mismatch");
            }
            if (!topology_switch_armed_.load() || prefix_topology_phase_.load() == PrefixTopologyPhase::QUARANTINED) {
                return absl::CancelledError("parallel topology switch prepare cancelled or quarantined");
            }
            const auto pause_at_ms = topology_switch_pause_at_ms_.load();
            if (pause_at_ms <= 0 || autil::TimeUtility::currentTimeInMilliSeconds() >= pause_at_ms) {
                scheduler_->wakeupForTopologySwitch();
                if (pause_.load()) {
                    std::unique_lock<std::mutex> execution_lock(execution_mutex_, std::try_to_lock);
                    if (execution_lock.owns_lock()) {
                        cudaSyncAndCheck();
                        return absl::OkStatus();
                    }
                }
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return absl::DeadlineExceededError(
                "timeout waiting for scheduled pause and idle executor before topology switch");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void NormalEngine::resumeParallelTopologySwitch() {
    if (prefix_topology_phase_.load() != PrefixTopologyPhase::NONE) {
        RTP_LLM_LOG_WARNING("refusing uncoordinated resume of a prefix topology transition");
        return;
    }
    topology_switch_armed_.store(false);
    topology_switch_pause_at_ms_.store(0);
    topology_switch_resume_step_count_.store(topology_switch_step_count_.load());
    if (executor_) {
        executor_->clearTopologySwitchPauseRequested();
    }
    pause_ = false;
    if (scheduler_) {
        scheduler_->wakeupForTopologySwitch();
    }
}

absl::Status NormalEngine::settleParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id) {
    const auto begin_time_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
    const auto effective_timeout = timeout_ms > 0 ? timeout_ms : 300000;
    const auto resume_step_count = topology_switch_resume_step_count_.load();
    while (true) {
        std::unique_lock<std::mutex> switch_lock(topology_switch_mutex_);
        if (resource_context_.cache_manager->prefixInventoryEnabled()
            && (switch_id.empty() || switch_id != last_topology_switch_id_ || !topology_switch_id_.empty()
                || prefix_topology_phase_.load() != PrefixTopologyPhase::NONE)) {
            return absl::FailedPreconditionError("settle topology identity is not the matching completed switch");
        }
        if (scheduler_) {
            scheduler_->wakeupForTopologySwitch();
        }
        if (!pause_.load() && !topology_switch_armed_.load()
            && (resume_step_count < 0 || topology_switch_step_count_.load() > resume_step_count)) {
            RTP_LLM_LOG_INFO("parallel topology switch settle done, wait_ms=%ld resume_step_count=%ld step_count=%ld",
                             autil::TimeUtility::currentTimeInMilliSeconds() - begin_time_ms,
                             resume_step_count,
                             topology_switch_step_count_.load());
            return absl::OkStatus();
        }
        if (autil::TimeUtility::currentTimeInMilliSeconds() - begin_time_ms > effective_timeout) {
            return absl::DeadlineExceededError("timeout waiting engine to settle after topology switch");
        }
        switch_lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

absl::Status NormalEngine::flushParallelTopologySwitch(int64_t timeout_ms, const std::string& switch_id) {
    const auto begin_time_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
    const auto effective_timeout = timeout_ms > 0 ? timeout_ms : 300000;
    const auto flush_step_count  = topology_switch_step_count_.load();
    while (true) {
        std::unique_lock<std::mutex> switch_lock(topology_switch_mutex_);
        if (resource_context_.cache_manager->prefixInventoryEnabled()
            && (switch_id.empty() || switch_id != topology_switch_id_
                || prefix_topology_phase_.load() != PrefixTopologyPhase::NONE)) {
            return absl::FailedPreconditionError("flush topology switch identity mismatch");
        }
        if (pause_.load() || topology_switch_armed_.load()) {
            return absl::FailedPreconditionError("cannot flush while topology switch is paused or armed");
        }
        if (scheduler_) {
            scheduler_->wakeupForTopologySwitch();
        }
        if (topology_switch_step_count_.load() > flush_step_count) {
            RTP_LLM_LOG_INFO("parallel topology switch flush done, wait_ms=%ld flush_step_count=%ld step_count=%ld",
                             autil::TimeUtility::currentTimeInMilliSeconds() - begin_time_ms,
                             flush_step_count,
                             topology_switch_step_count_.load());
            return absl::OkStatus();
        }
        if (autil::TimeUtility::currentTimeInMilliSeconds() - begin_time_ms > effective_timeout) {
            return absl::DeadlineExceededError("timeout waiting engine to flush before topology switch");
        }
        switch_lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

absl::Status NormalEngine::preparePrefixTopologySwitch(int64_t            tp_size,
                                                       int64_t            dp_size,
                                                       int64_t            timeout_ms,
                                                       const std::string& switch_id) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 300000);
    while (true) {
        {
            std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
            if (switch_id != topology_switch_id_ || !topology_switch_armed_.load() || !pause_.load()) {
                return absl::FailedPreconditionError("prefix prepare requires the matching paused topology switch");
            }
            if (!resource_context_.cache_manager->prefixInventoryEnabled()
                || (parallelism_config.tp_size == tp_size && parallelism_config.dp_size == dp_size)) {
                return absl::OkStatus();
            }
            if (switch_id.empty()) {
                return absl::InvalidArgumentError("prefix topology switch identity is required");
            }
            if (prefix_topology_phase_.load() != PrefixTopologyPhase::NONE
                && (prefix_topology_phase_.load() != PrefixTopologyPhase::PREPARED
                    || prefix_topology_target_tp_ != tp_size || prefix_topology_target_dp_ != dp_size)) {
                return absl::FailedPreconditionError("prefix topology prepare conflicts with current transition");
            }
            std::unique_lock<std::mutex> execution_lock(execution_mutex_, std::try_to_lock);
            if (execution_lock.owns_lock()) {
                try {
                    (void)makeSwitchedParallelismConfig(parallelism_config, tp_size, dp_size);
                    if (prefix_topology_phase_.load() == PrefixTopologyPhase::NONE) {
                        prefix_topology_generation_ = resource_context_.cache_manager->beginPrefixTopologyChange();
                        prefix_topology_target_tp_  = tp_size;
                        prefix_topology_target_dp_  = dp_size;
                        prefix_topology_phase_.store(PrefixTopologyPhase::PREPARED);
                    }
                    if (resource_context_.cache_manager->prefixTopologyChangeDrained(prefix_topology_generation_)) {
                        return absl::OkStatus();
                    }
                } catch (const std::exception& error) {
                    return absl::FailedPreconditionError(error.what());
                }
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return absl::DeadlineExceededError("timeout draining physical prefix transfers before topology switch");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

absl::Status NormalEngine::prepareResidentPrefixTopologySwitch(int64_t            tp_size,
                                                               int64_t            dp_size,
                                                               const std::string& switch_id,
                                                               bool               leader_merge) {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (switch_id.empty() || switch_id != topology_switch_id_ || !topology_switch_armed_.load() || !pause_.load()
        || prefix_topology_phase_.load() != PrefixTopologyPhase::PREPARED || prefix_topology_target_tp_ != tp_size
        || prefix_topology_target_dp_ != dp_size) {
        return absl::FailedPreconditionError("resident prefix prepare requires the matching prepared transition");
    }
    std::lock_guard<std::mutex> execution_lock(execution_mutex_);
    try {
        const auto target = makeSwitchedParallelismConfig(parallelism_config, tp_size, dp_size);
        const bool prepared =
            leader_merge ?
                resource_context_.cache_manager->prepareResidentPrefixMerge(prefix_topology_generation_, target) :
                resource_context_.cache_manager->prepareResidentPrefixSplit(prefix_topology_generation_, target);
        if (!prepared) {
            return absl::FailedPreconditionError("resident prefix preparation failed");
        }
    } catch (const std::exception& error) {
        return absl::FailedPreconditionError(error.what());
    }
    return absl::OkStatus();
}

void NormalEngine::quarantinePrefixTopologyLocked() {
    pause_ = true;
    scheduler_->setDrainMode(true);
    prefix_topology_phase_.store(PrefixTopologyPhase::QUARANTINED);
    try {
        resource_context_.cache_manager->beginPrefixTopologyChange();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("prefix topology quarantine: %s", error.what());
    }
    resource_context_.cache_manager->fencePrefixInventory();
}

absl::Status NormalEngine::quarantineParallelTopologySwitch(const std::string& switch_id) {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (switch_id.empty()) {
        return absl::InvalidArgumentError("quarantine topology switch identity is required");
    }
    if (switch_id != topology_switch_id_ && !(topology_switch_id_.empty() && switch_id == last_topology_switch_id_)) {
        return absl::FailedPreconditionError("quarantine topology switch identity mismatch");
    }
    quarantinePrefixTopologyLocked();
    return absl::OkStatus();
}

absl::Status NormalEngine::finishParallelTopologySwitch(bool cancel, const std::string& switch_id) {
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (switch_id.empty() && resource_context_.cache_manager->prefixInventoryEnabled()) {
        return absl::InvalidArgumentError("finish prefix topology switch identity is required");
    }
    if (topology_switch_id_.empty() && !switch_id.empty() && switch_id == last_topology_switch_id_
        && prefix_topology_phase_.load() == PrefixTopologyPhase::NONE) {
        return absl::OkStatus();
    }
    if (switch_id != topology_switch_id_) {
        return absl::FailedPreconditionError("finish topology switch identity mismatch");
    }
    const auto phase = prefix_topology_phase_.load();
    if (phase == PrefixTopologyPhase::QUARANTINED) {
        return absl::FailedPreconditionError("topology switch remains quarantined");
    }
    if (cancel && phase == PrefixTopologyPhase::COMMITTED) {
        quarantinePrefixTopologyLocked();
        return absl::FailedPreconditionError("cannot cancel after communicator mutation");
    }
    if (!cancel && phase == PrefixTopologyPhase::PREPARED) {
        return absl::FailedPreconditionError("cannot resume before topology commit");
    }
    std::lock_guard<std::mutex> execution_lock(execution_mutex_);
    if (phase == PrefixTopologyPhase::PREPARED
        && !resource_context_.cache_manager->cancelPrefixTopologyChange(prefix_topology_generation_)) {
        return absl::UnavailableError("prefix topology cancellation is waiting for physical drain");
    }
    if (phase == PrefixTopologyPhase::COMMITTED) {
        try {
            if (!resource_context_.cache_manager->completePrefixTopologyChange(prefix_topology_generation_,
                                                                               parallelism_config)) {
                quarantinePrefixTopologyLocked();
                return absl::InternalError("failed to install prefix topology configuration");
            }
        } catch (const std::exception& error) {
            quarantinePrefixTopologyLocked();
            return absl::InternalError(error.what());
        }
    }
    prefix_topology_phase_.store(PrefixTopologyPhase::NONE);
    prefix_topology_generation_ = 0;
    last_topology_switch_id_    = topology_switch_id_;
    topology_switch_id_.clear();
    resumeParallelTopologySwitch();
    if (cancel) {
        scheduler_->setDrainMode(false);
    }
    return absl::OkStatus();
}

absl::Status NormalEngine::switchParallelTopology(int64_t            tp_size,
                                                  int64_t            dp_size,
                                                  int64_t            timeout_ms,
                                                  const std::string& switch_id) {
    (void)timeout_ms;
    std::lock_guard<std::mutex> switch_lock(topology_switch_mutex_);
    if (prefix_topology_phase_.load() == PrefixTopologyPhase::QUARANTINED
        || (!topology_switch_id_.empty() && switch_id != topology_switch_id_)) {
        return absl::FailedPreconditionError("topology commit identity mismatch or quarantined");
    }
    if (parallelism_config.tp_size == tp_size && parallelism_config.dp_size == dp_size) {
        RTP_LLM_LOG_INFO("parallel topology switch no-op: tp_size=%ld dp_size=%ld", tp_size, dp_size);
        return absl::OkStatus();
    }

    RTP_LLM_LOG_INFO(
        "parallel topology switch begin: %s -> TP%ldDP%ld", parallelism_config.to_string().c_str(), tp_size, dp_size);
    ParallelismConfig new_config;
    try {
        new_config = makeSwitchedParallelismConfig(parallelism_config, tp_size, dp_size);
    } catch (const std::exception& e) {
        resumeParallelTopologySwitch();
        return absl::InvalidArgumentError(e.what());
    }

    if (resource_context_.cache_manager->prefixInventoryEnabled()
        && (prefix_topology_phase_.load() != PrefixTopologyPhase::PREPARED || prefix_topology_target_tp_ != tp_size
            || prefix_topology_target_dp_ != dp_size)) {
        return absl::FailedPreconditionError("topology commit requires the matching prefix prepare");
    }

    pause_      = true;
    auto resume = [this]() { resumeParallelTopologySwitch(); };
    try {
        std::lock_guard<std::mutex> execution_lock(execution_mutex_);
        const bool                  prefix_enabled = resource_context_.cache_manager->prefixInventoryEnabled();
        if (prefix_enabled) {
            if (prefix_topology_phase_.load() != PrefixTopologyPhase::PREPARED || prefix_topology_target_tp_ != tp_size
                || prefix_topology_target_dp_ != dp_size
                || !resource_context_.cache_manager->prefixTopologyChangeDrained(prefix_topology_generation_)) {
                return absl::FailedPreconditionError("topology commit requires the prepared and drained prefix gate");
            }
            prefix_topology_phase_.store(PrefixTopologyPhase::COMMITTED);
        } else {
            resource_context_.cache_manager->fencePrefixInventory();
        }
        {
            pybind11::gil_scoped_acquire gil;
            auto collective_module = pybind11::module_::import("rtp_llm.models_py.distributed.collective_torch");
            collective_module.attr("switch_parallel_topology")(tp_size, dp_size);
        }
        if (executor_ && !executor_->updateParallelismConfig(new_config)) {
            if (prefix_enabled) {
                quarantinePrefixTopologyLocked();
            } else {
                resume();
            }
            return absl::InternalError("executor failed to update parallelism config");
        }
        parallelism_config = new_config;
        scheduler_->updateParallelismConfig(parallelism_config);
        RTP_LLM_LOG_INFO("parallel topology switch done: %s", parallelism_config.to_string().c_str());
    } catch (const pybind11::error_already_set& e) {
        if (prefix_topology_phase_.load() != PrefixTopologyPhase::NONE) {
            quarantinePrefixTopologyLocked();
        } else {
            resume();
        }
        return absl::InternalError(std::string("python switch_parallel_topology failed: ") + e.what());
    } catch (const std::exception& e) {
        if (prefix_topology_phase_.load() != PrefixTopologyPhase::NONE) {
            quarantinePrefixTopologyLocked();
        } else {
            resume();
        }
        return absl::InternalError(std::string("switchParallelTopology failed: ") + e.what());
    }
    return absl::OkStatus();
}

void NormalEngine::startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {
    step_profiler_.configure(true, trace_name, start_step, num_steps);
}

bool NormalEngine::isMTPEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_MTP || propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

bool NormalEngine::isEagle() {
    if (propose_params_) {
        return propose_params_->sp_type == SP_TYPE_EAGLE;
    }
    return false;
}

void NormalEngine::mayAddFakeStream(std::list<GenerateStreamPtr>& streams) {
    if (isMTPEagle()) {
        int propose_step = sp_config.gen_num_per_cycle;
        switch (pd_sep_config.role_type) {
            case RoleType::PREFILL:
                if (streams.empty()) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                break;
            case RoleType::DECODE:
                if (streams.empty()) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_));
                }
                break;
            case RoleType::PDFUSION: {
                bool has_prefill = false;
                bool has_decode  = false;
                for (auto& stream : streams) {
                    if (stream->isContextStream()) {
                        has_prefill = true;
                    } else {
                        has_decode = true;
                    }
                }
                if (!has_prefill) {
                    streams.emplace_back(
                        MtpExecutor::createMinFakePrefillStream(1, model_config_, runtime_config, resource_context_));
                }
                if (!has_decode) {
                    streams.emplace_back(MtpExecutor::createMinFakeDecodeStream(
                        propose_step, model_config_, runtime_config, resource_context_));
                }
                break;
            }
            default:
                RTP_LLM_CHECK_WITH_INFO(false, "invalid role type");
                break;
        }
    } else {
        if (streams.empty()) {
            streams.emplace_back(createMinFakeStream(1));
        }
    }
}

}  // namespace rtp_llm
