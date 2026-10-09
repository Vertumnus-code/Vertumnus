#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/cache/PrefixCacheTransactionManager.h"

namespace rtp_llm {

struct PrefixTopologyController: public autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper& json) override {
        json.Jsonize("namespace", namespace_id, std::string(""));
        json.Jsonize("controller_id", controller_id, std::string(""));
        json.Jsonize("term", term, uint64_t(0));
    }

    PrefixControllerIdentity identity() const {
        return {namespace_id, controller_id, term};
    }

    std::string namespace_id;
    std::string controller_id;
    uint64_t    term = 0;
};

struct PrefixTopologyParticipant: public autil::legacy::Jsonizable {
    void Jsonize(JsonWrapper& json) override {
        json.Jsonize("boot_id", boot_id, std::string(""));
        json.Jsonize("epoch", epoch, std::string(""));
        json.Jsonize("world_rank", world_rank, int64_t(-1));
    }

    std::string boot_id;
    std::string epoch;
    int64_t     world_rank = -1;
};

}  // namespace rtp_llm
