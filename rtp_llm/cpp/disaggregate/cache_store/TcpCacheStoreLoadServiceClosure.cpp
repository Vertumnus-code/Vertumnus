#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <atomic>
#include <cstring>

namespace rtp_llm {

namespace {
std::atomic<int64_t> cache_load_diagnostic_seq{0};
constexpr int64_t    cache_load_diagnostic_limit = 4096;
}  // namespace

TcpCacheStoreLoadServiceClosure::~TcpCacheStoreLoadServiceClosure() {
    if (controller_) {
        delete controller_;
    }
    if (request_) {
        delete request_;
    }
    if (response_) {
        delete response_;
    }
}

void TcpCacheStoreLoadServiceClosure::Run() {
    diagnostic_seq_ = cache_load_diagnostic_seq.fetch_add(1, std::memory_order_relaxed);
    if (diagnostic_seq_ < cache_load_diagnostic_limit) {
        RTP_LLM_LOG_INFO("cache load response callback begin, seq %ld, request %s, controller failed %d, blocks %d",
                         diagnostic_seq_,
                         request_block_buffer_->getRequestKey().c_str(),
                         controller_->Failed(),
                         response_->blocks_size());
    }
    collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());
    if (!copy_thread_pool_) {
        runImpl();
        return;
    }

    auto ret = copy_thread_pool_->pushTask([this]() { this->runImpl(); });
    if (diagnostic_seq_ < cache_load_diagnostic_limit) {
        RTP_LLM_LOG_INFO("cache load response callback queued, seq %ld, request %s, result %d",
                         diagnostic_seq_,
                         request_block_buffer_->getRequestKey().c_str(),
                         static_cast<int>(ret));
    }
    if (ret != autil::ThreadPoolBase::ERROR_NONE) {
        RTP_LLM_LOG_WARNING("cache load response copy queue is full");
        end(false, CacheStoreErrorCode::PushWorkerItemFailed);
    }
}

void TcpCacheStoreLoadServiceClosure::runImpl() {
    if (diagnostic_seq_ < cache_load_diagnostic_limit) {
        RTP_LLM_LOG_INFO("cache load response copy begin, seq %ld, request %s",
                         diagnostic_seq_,
                         request_block_buffer_->getRequestKey().c_str());
    }
    pinThreadToDeviceOnce(device_id_);

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("cache load request failed, controller err is %d", controller_->GetErrorCode());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("cache load request failed, response err is %d", response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            RTP_LLM_LOG_WARNING("can not find match block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        if (unload_block->gpu_mem) {
#if USING_CUDA
            const auto rc = cudaMemcpy(
                unload_block->addr.get(), block.content().data(), block.len(), cudaMemcpyHostToDevice);
            if (rc != cudaSuccess) {
                RTP_LLM_LOG_WARNING("cache load H2D copy failed for request %s: %s",
                                    request_block_buffer_->getRequestKey().c_str(),
                                    cudaGetErrorString(rc));
                end(false, CacheStoreErrorCode::LoadErrorUnknown);
                return;
            }
#elif USING_ROCM
            const auto rc = hipMemcpy(
                unload_block->addr.get(), block.content().data(), block.len(), hipMemcpyHostToDevice);
            if (rc != hipSuccess) {
                RTP_LLM_LOG_WARNING("cache load H2D copy failed for request %s: %s",
                                    request_block_buffer_->getRequestKey().c_str(),
                                    hipGetErrorString(rc));
                end(false, CacheStoreErrorCode::LoadErrorUnknown);
                return;
            }
#else
            std::memcpy(unload_block->addr.get(), block.content().data(), block.len());
#endif
        } else {
            std::memcpy(unload_block->addr.get(), block.content().data(), block.len());
        }
    }
    if (diagnostic_seq_ < cache_load_diagnostic_limit) {
        RTP_LLM_LOG_INFO("cache load response copy end, seq %ld, request %s",
                         diagnostic_seq_,
                         request_block_buffer_->getRequestKey().c_str());
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    callback_(success, ec);
    delete this;
}

}  // namespace rtp_llm
