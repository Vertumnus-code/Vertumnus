package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CacheStatus {
    @JsonProperty("inventory")
    private org.flexlb.dao.cache.CacheReportV2 inventory;

    @JsonProperty("message")
    private String message;
    @JsonProperty("available_kv_cache")
    private long availableKvCache;  // available kv cache tokens
    @JsonProperty("total_kv_cache")
    private long totalKvCache;      // total kv cache tokens
    @JsonProperty("block_size")
    private long blockSize;
    @JsonProperty("version")
    private long version = -1;
    @JsonProperty("cached_keys")
    private Set<Long> cachedKeys;
    @JsonProperty("cache_key_size")
    private long cacheKeySize;
    @JsonProperty("address")
    private String address;
    @JsonProperty("dp_size")
    private long dpSize;
    @JsonProperty("results")
    private List<CacheStatus> results;

    /**
     * The engine's HTTP {@code /cache_status} reports residency as a hash-to-flag
     * object rather than a list, so accept that shape into {@link #cachedKeys}.
     */
    @JsonProperty("cache_keys")
    void setCacheKeysMap(Map<String, Boolean> cacheKeys) {
        if (cacheKeys == null) {
            return;
        }
        Set<Long> keys = new HashSet<>(cacheKeys.size());
        for (Map.Entry<String, Boolean> entry : cacheKeys.entrySet()) {
            if (!Boolean.TRUE.equals(entry.getValue())) {
                continue;
            }
            try {
                keys.add(Long.parseLong(entry.getKey()));
            } catch (NumberFormatException ignored) {
                // a non-numeric hash carries no block identity
            }
        }
        this.cachedKeys = keys;
        this.cacheKeySize = keys.size();
    }
}
