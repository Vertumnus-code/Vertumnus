package org.flexlb.cache.prefix;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

final class PrefixPath extends AbstractList<Long> {
    private final PrefixPath parent;
    private final Long block;
    private final int length;
    private volatile Long[] materialized;

    PrefixPath(PrefixPath parent, long block) {
        this.parent = parent;
        this.block = block;
        this.length = parent == null ? 1 : Math.addExact(parent.size(), 1);
    }

    PrefixPath parent() { return parent; }
    Long block() { return block; }
    @Override public int size() { return length; }
    @Override public Long get(int index) {
        Objects.checkIndex(index, length);
        return values()[index];
    }
    @Override public Iterator<Long> iterator() { return Arrays.asList(values()).iterator(); }
    @Override public Object[] toArray() { return values().clone(); }

    private Long[] values() {
        var result = materialized;
        if (result == null) {
            result = new Long[length];
            PrefixPath node = this;
            for (int index = length - 1; index >= 0; index--) {
                result[index] = node.block;
                node = node.parent;
            }
            materialized = result;
        }
        return result;
    }
}
