package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineRpcService.PrefixCacheCapabilitiesResponsePB;
import org.flexlb.engine.grpc.EngineRpcService.PrefixReplicatedLayoutPB;

public final class PrefixLayoutCompatibility {
    private PrefixLayoutCompatibility() { }

    public static long requireCompatible(PrefixCacheCapabilitiesResponsePB source,
                                          PrefixCacheCapabilitiesResponsePB destination, String namespace) {
        validate(source, namespace);
        validate(destination, namespace);
        if (!source.getReplicatedLayout().equals(destination.getReplicatedLayout())) {
            throw new IllegalArgumentException("prefix KV geometry or attention semantics are incompatible");
        }
        return source.getReplicatedLayout().getBytesPerRankBlock();
    }

    public static void validate(PrefixCacheCapabilitiesResponsePB capability, String namespace) {
        if (namespace == null || namespace.isBlank() || !namespace.equals(capability.getNamespace())
                || !capability.getDeviceReady() || capability.getDpLane() < 0 || capability.getCpDegree() <= 0
                || capability.getTpSize() != capability.getCpDegree() || capability.getLayoutFingerprint().isBlank()
                || !capability.getSupportedOperationsList().contains("replicated_mha_layout_v1")
                || !capability.hasReplicatedLayout()) {
            throw new IllegalArgumentException("missing verified replicated prefix layout capability");
        }
        PrefixReplicatedLayoutPB layout = capability.getReplicatedLayout();
        if (layout.getSchemaVersion() != 1
                || !layout.getStorageKind().equals("REPLICATED_FULL_MHA_LAYER_MERGED_KV_V1")
                || !layout.getUnknownFields().asMap().isEmpty() || !layout.hasRope()
                || !layout.getRope().getUnknownFields().asMap().isEmpty()
                || layout.getGlobalLayerIdsCount() < 1 || layout.getGlobalLayerIdsCount() > 4096
                || layout.getGlobalQueryHeads() <= 0 || layout.getGlobalQueryHeads() > 65536
                || layout.getGlobalKvHeads() <= 0 || layout.getGlobalKvHeads() > 65536
                || layout.getGlobalQueryHeads() % layout.getGlobalKvHeads() != 0
                || layout.getHeadDimension() <= 0 || layout.getHeadDimension() > 4096
                || layout.getTokensPerBlock() <= 0 || layout.getTokensPerBlock() > 1048576
                || layout.getTokensPerBlock() != capability.getBlockSize()
                || layout.getTokensPerBlock() != layout.getKernelTokensPerBlock()) {
            throw new IllegalArgumentException("unsupported or malformed replicated prefix geometry");
        }
        int itemBytes;
        int quantization;
        switch (layout.getNativeDtype()) {
            case 10, 14 -> { itemBytes = 2; quantization = 0; }
            case 6 -> { itemBytes = 1; quantization = 1; }
            case 15 -> { itemBytes = 1; quantization = 2; }
            default -> throw new IllegalArgumentException("unsupported prefix KV scalar type");
        }
        if (layout.getKvQuantization() != quantization) {
            throw new IllegalArgumentException("prefix KV quantization and scalar type disagree");
        }
        long kvBytes = Math.multiplyExact(Math.multiplyExact(Math.multiplyExact(2, layout.getGlobalKvHeads()),
                layout.getHeadDimension()), Math.multiplyExact(layout.getTokensPerBlock(), itemBytes));
        long scaleBytes = quantization == 0 ? 0 : Math.multiplyExact(
                Math.multiplyExact(2, layout.getGlobalKvHeads()), Math.multiplyExact(layout.getTokensPerBlock(), 4));
        int buffersPerLayer = scaleBytes == 0 ? 1 : 2;
        if (layout.getKvBytesPerLayer() != kvBytes || layout.getScaleBytesPerLayer() != scaleBytes
                || layout.getBytesPerRankBlock() != Math.multiplyExact(Math.addExact(kvBytes, scaleBytes), layout.getGlobalLayerIdsCount())
                || layout.getBufferBytesCount() != layout.getGlobalLayerIdsCount() * buffersPerLayer) {
            throw new IllegalArgumentException("replicated prefix byte footprint is inconsistent");
        }
        for (int layer = 0; layer < layout.getGlobalLayerIdsCount(); layer++) {
            if (layout.getGlobalLayerIds(layer) != layer || layout.getBufferBytes(layer * buffersPerLayer) != kvBytes
                    || (scaleBytes > 0 && layout.getBufferBytes(layer * buffersPerLayer + 1) != scaleBytes)) {
                throw new IllegalArgumentException("prefix layer coverage or KV/scale buffer order is invalid");
            }
        }
        var rope = layout.getRope();
        if (!Double.isFinite(layout.getLayernormEpsilon()) || !Double.isFinite(layout.getPartialRotaryFactor())
                || !Float.isFinite(layout.getQScaling()) || !Float.isFinite(layout.getSoftmaxExtraScale())
                || !Float.isFinite(rope.getScale()) || !Float.isFinite(rope.getFactor1()) || !Float.isFinite(rope.getFactor2())
                || !Float.isFinite(rope.getExtrapolationFactor()) || !Float.isFinite(rope.getMscale())) {
            throw new IllegalArgumentException("nonfinite prefix attention semantics");
        }
    }
}
