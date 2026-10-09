package org.flexlb.cache.prefix;

public record PrefillCostModel(double floor, double fixed, double shard, double linear,
                                double cache, double attention) {
    public double seconds(double cpSize, double uncachedLen, double prefixLen) {
        double remaining = Math.max(0.0, uncachedLen);
        double prefix = Math.max(0.0, prefixLen);
        double remainingK = remaining / 1024.0;
        double prefixK = prefix / 1024.0;
        double workK = (remaining * prefix + remaining * (remaining + 1.0) / 2.0) / (1024.0 * 1024.0);
        return Math.max(floor, fixed
                + (shard / cpSize + linear) * remainingK
                + cache * prefixK
                + (attention / cpSize) * workK);
    }

    public static long maxReusableTokens(long inputTokens, long blockSize) {
        if (blockSize <= 0) { throw new IllegalArgumentException("block size must be positive"); }
        return inputTokens <= 0 ? 0 : ((inputTokens - 1) / blockSize) * blockSize;
    }
}
