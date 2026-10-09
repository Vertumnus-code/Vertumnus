package org.flexlb.sync.prefix;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

final class PrefixTopologyJournalCodec {
    static final int MAX_STORED_BYTES = 524288;
    static final int MAX_JSON_BYTES = 8 * 1024 * 1024;

    private PrefixTopologyJournalCodec() { }

    static byte[] encode(byte[] json) throws IOException {
        if (json.length > MAX_JSON_BYTES) { throw new IllegalArgumentException("oversized topology JSON"); }
        if (json.length <= MAX_STORED_BYTES) { return json; }
        var output = new ByteArrayOutputStream();
        try (var gzip = new GZIPOutputStream(output)) { gzip.write(json); }
        byte[] data = output.toByteArray();
        if (data.length > MAX_STORED_BYTES) { throw new IllegalArgumentException("oversized compressed topology plan"); }
        return data;
    }

    static byte[] decode(byte[] data) throws IOException {
        if (data.length > MAX_STORED_BYTES) { throw new IllegalArgumentException("oversized topology entry"); }
        // Existing schema-1 JSON remains readable; gzip only changes storage encoding.
        if (data.length < 2 || data[0] != (byte) 0x1f || data[1] != (byte) 0x8b) { return data; }
        try (var gzip = new GZIPInputStream(new ByteArrayInputStream(data))) {
            byte[] json = gzip.readNBytes(MAX_JSON_BYTES + 1);
            if (json.length > MAX_JSON_BYTES) { throw new IllegalArgumentException("oversized topology JSON"); }
            return json;
        }
    }
}
