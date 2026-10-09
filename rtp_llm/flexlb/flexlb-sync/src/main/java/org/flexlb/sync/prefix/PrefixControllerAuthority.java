package org.flexlb.sync.prefix;

import org.flexlb.engine.grpc.EngineRpcService;

public interface PrefixControllerAuthority {
    EngineRpcService.PrefixControllerPB snapshot(String namespace);

    boolean isCurrent(EngineRpcService.PrefixControllerPB controller);
}
