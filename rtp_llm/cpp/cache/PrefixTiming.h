#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <utility>
#include <unistd.h>

namespace rtp_llm {

class PrefixTiming {
public:
    enum class Phase {
        Commit,
        Validation,
        ReceiveWait,
        RankWait,
        Publication,
        RankInstall
    };

    struct Timestamp {
        int64_t monotonic_ns;
        int64_t unix_ms;
    };

    struct Sample {
        Phase     phase;
        uint64_t  operation_id;
        int       rank;
        Timestamp started;
        int64_t   elapsed_ns;
    };

    using Clock = Timestamp (*)() noexcept;
    using Sink  = void (*)(const Sample&);

    static bool configured(const char* value) noexcept {
        return value != nullptr && std::strcmp(value, "1") == 0;
    }

    static bool enabled() noexcept {
        static const bool value = configured(std::getenv("RTP_LLM_PREFIX_NATIVE_TIMING"));
        return value;
    }

    explicit PrefixTiming(Phase phase, uint64_t operation_id, int rank = -1):
        PrefixTiming(phase, operation_id, rank, enabled(), now, emit) {}

    PrefixTiming(Phase phase, uint64_t operation_id, int rank, bool active):
        PrefixTiming(phase, operation_id, rank, active, now, emit) {}

    PrefixTiming(Phase phase, uint64_t operation_id, int rank, bool active, Clock clock, Sink sink):
        active_(active), clock_(clock), sink_(sink), sample_{phase, operation_id, rank, {}, 0} {
        if (active_) {
            sample_.started = clock_();
        }
    }

    ~PrefixTiming() noexcept {
        finish();
    }

    PrefixTiming(const PrefixTiming&)            = delete;
    PrefixTiming& operator=(const PrefixTiming&) = delete;

    void finish() noexcept {
        if (!active_) {
            return;
        }
        active_            = false;
        sample_.elapsed_ns = clock_().monotonic_ns - sample_.started.monotonic_ns;
        try {
            sink_(sample_);
        } catch (...) {}
    }

    template<typename Callback>
    static decltype(auto) measure(Phase phase, uint64_t operation_id, Callback&& callback) {
        PrefixTiming timing(phase, operation_id);
        return std::forward<Callback>(callback)();
    }

private:
    static Timestamp now() noexcept {
        return {
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count()};
    }

    static const char* name(Phase phase) noexcept {
        switch (phase) {
            case Phase::Commit:
                return "commit";
            case Phase::Validation:
                return "validation";
            case Phase::ReceiveWait:
                return "receive_wait";
            case Phase::RankWait:
                return "rank_wait";
            case Phase::Publication:
                return "publication";
            case Phase::RankInstall:
                return "rank_install";
        }
        return "unknown";
    }

    static void emit(const Sample& sample) {
        static std::atomic<uint64_t> sequence{0};
        std::fprintf(stderr,
                     "PREFIX_NATIVE_TIMING {\"schema\":1,\"sequence\":%llu,\"pid\":%ld,\"phase\":\"%s\","
                     "\"operation_id\":%llu,\"rank\":%d,\"started_unix_ms\":%lld,\"started_ns\":%lld,"
                     "\"elapsed_ns\":%lld}\n",
                     static_cast<unsigned long long>(sequence.fetch_add(1, std::memory_order_relaxed) + 1),
                     static_cast<long>(getpid()),
                     name(sample.phase),
                     static_cast<unsigned long long>(sample.operation_id),
                     sample.rank,
                     static_cast<long long>(sample.started.unix_ms),
                     static_cast<long long>(sample.started.monotonic_ns),
                     static_cast<long long>(sample.elapsed_ns));
    }

    bool   active_;
    Clock  clock_;
    Sink   sink_;
    Sample sample_;
};

}  // namespace rtp_llm
