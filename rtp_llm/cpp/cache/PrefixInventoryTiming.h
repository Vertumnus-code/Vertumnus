#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace rtp_llm {

class PrefixInventoryTiming {
public:
    enum class Boundary {
        PreForward,
        PostForward,
        Idle
    };

    struct Timestamp {
        int64_t monotonic_ns;
        int64_t unix_ms;
    };

    struct Sample {
        Boundary               boundary;
        int                    rank;
        int                    rank_count;
        Timestamp              started;
        std::array<int64_t, 4> stage_ns;
        int64_t                elapsed_ns;
        bool                   all_ranks_ready;
        bool                   complete;
    };

    using Clock = Timestamp (*)() noexcept;
    using Sink  = void (*)(const Sample&);

    static bool configured(const char* value) noexcept {
        return value && std::strcmp(value, "1") == 0;
    }

    static bool enabled() noexcept {
        static const bool value = configured(std::getenv("RTP_LLM_PREFIX_INVENTORY_TIMING"));
        return value;
    }

    PrefixInventoryTiming(Boundary boundary, int rank, int rank_count):
        PrefixInventoryTiming(boundary, rank, rank_count, enabled(), now, emit) {}

    PrefixInventoryTiming(Boundary boundary, int rank, int rank_count, bool active, Clock clock, Sink sink):
        active_(active),
        clock_(clock),
        sink_(sink),
        sample_{boundary, rank, rank_count, {}, {-1, -1, -1, -1}, 0, false, false} {
        if (active_) {
            sample_.started = clock_();
            last_ns_        = sample_.started.monotonic_ns;
        }
    }

    ~PrefixInventoryTiming() noexcept {
        finish();
    }

    PrefixInventoryTiming(const PrefixInventoryTiming&)            = delete;
    PrefixInventoryTiming& operator=(const PrefixInventoryTiming&) = delete;

    void snapshotCaptured() noexcept {
        mark(0);
    }
    void streamSynchronized() noexcept {
        mark(1);
    }
    void ranksReady(bool ready) noexcept {
        if (active_) {
            sample_.all_ranks_ready = ready;
        }
        mark(2);
    }
    void confirmed() noexcept {
        mark(3);
    }

    void finish() noexcept {
        if (!active_) {
            return;
        }
        active_            = false;
        sample_.elapsed_ns = clock_().monotonic_ns - sample_.started.monotonic_ns;
        sample_.complete   = next_stage_ == sample_.stage_ns.size() && valid_;
        try {
            sink_(sample_);
        } catch (...) {}
    }

private:
    void mark(size_t stage) noexcept {
        if (!active_) {
            return;
        }
        if (stage != next_stage_) {
            valid_ = false;
            return;
        }
        const auto captured     = clock_().monotonic_ns;
        sample_.stage_ns[stage] = captured - last_ns_;
        last_ns_                = captured;
        ++next_stage_;
    }

    static Timestamp now() noexcept {
        return {
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count()};
    }

    static const char* name(Boundary boundary) noexcept {
        switch (boundary) {
            case Boundary::PreForward:
                return "pre_forward";
            case Boundary::PostForward:
                return "post_forward";
            case Boundary::Idle:
                return "idle";
        }
        return "unknown";
    }

    static void emit(const Sample& sample) {
        static std::atomic<uint64_t> sequence{0};
        std::fprintf(stderr,
                     "PREFIX_INVENTORY_TIMING {\"schema\":1,\"sequence\":%llu,\"pid\":%ld,\"boundary\":\"%s\","
                     "\"rank\":%d,\"rank_count\":%d,\"started_unix_ms\":%lld,\"started_ns\":%lld,"
                     "\"snapshot_ns\":%lld,\"stream_sync_ns\":%lld,\"all_rank_ready_ns\":%lld,"
                     "\"confirmation_ns\":%lld,\"elapsed_ns\":%lld,\"all_ranks_ready\":%s,\"complete\":%s}\n",
                     static_cast<unsigned long long>(sequence.fetch_add(1, std::memory_order_relaxed) + 1),
                     static_cast<long>(getpid()),
                     name(sample.boundary),
                     sample.rank,
                     sample.rank_count,
                     static_cast<long long>(sample.started.unix_ms),
                     static_cast<long long>(sample.started.monotonic_ns),
                     static_cast<long long>(sample.stage_ns[0]),
                     static_cast<long long>(sample.stage_ns[1]),
                     static_cast<long long>(sample.stage_ns[2]),
                     static_cast<long long>(sample.stage_ns[3]),
                     static_cast<long long>(sample.elapsed_ns),
                     sample.all_ranks_ready ? "true" : "false",
                     sample.complete ? "true" : "false");
    }

    bool    active_;
    Clock   clock_;
    Sink    sink_;
    Sample  sample_;
    int64_t last_ns_    = 0;
    size_t  next_stage_ = 0;
    bool    valid_      = true;
};

}  // namespace rtp_llm
