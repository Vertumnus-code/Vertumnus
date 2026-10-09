# Bazel (https://bazel.build/) BUILD file for Protobuf.

licenses(["notice"])

exports_files(["LICENSE"])

################################################################################
# Java 9 configuration
################################################################################


################################################################################
# ZLIB configuration
################################################################################

#ZLIB_DEPS = ["@zlib//:zlib"]
ZLIB_DEPS = ["//external:zlib"]

################################################################################
# Protobuf Runtime Library
################################################################################

MSVC_COPTS = [
    "/DHAVE_PTHREAD",
    "/wd4018", # -Wno-sign-compare
    "/wd4065", # switch statement contains 'default' but no 'case' labels
    "/wd4146", # unary minus operator applied to unsigned type, result still unsigned
    "/wd4244", # 'conversion' conversion from 'type1' to 'type2', possible loss of data
    "/wd4251", # 'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'
    "/wd4267", # 'var' : conversion from 'size_t' to 'type', possible loss of data
    "/wd4305", # 'identifier' : truncation from 'type1' to 'type2'
    "/wd4307", # 'operator' : integral constant overflow
    "/wd4309", # 'conversion' : truncation of constant value
    "/wd4334", # 'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)
    "/wd4355", # 'this' : used in base member initializer list
    "/wd4506", # no definition for inline function 'function'
    "/wd4514", # -Wno-unused-function
    "/wd4800", # 'type' : forcing value to bool 'true' or 'false' (performance warning)
    "/wd4996", # The compiler encountered a deprecated declaration.
]

COPTS = select({
    ":msvc" : MSVC_COPTS,
    "//conditions:default": [
        "-DHAVE_PTHREAD",
        "-DHAVE_ZLIB",
        "-Woverloaded-virtual",
        "-Wno-sign-compare",
        "-Wno-unused-function",
        # Prevents ISO C++ const string assignment warnings for pyext sources.
        "-Wno-write-strings",
    ],
})

load(":compiler_config_setting.bzl", "create_compiler_config_setting")

create_compiler_config_setting(name = "msvc", value = "msvc-cl")

config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
)

config_setting(
    name = "android-libcpp",
    values = {
        "crosstool_top": "@androidndk//:toolchain-libcpp",
    },
)

config_setting(
    name = "android-gnu-libstdcpp",
    values = {
        "crosstool_top": "@androidndk//:toolchain-gnu-libstdcpp",
    },
)

# Android and MSVC builds do not need to link in a separate pthread library.
LINK_OPTS = select({
    ":android": [],
    ":android-libcpp": [],
    ":android-gnu-libstdcpp": [],
    ":msvc": [
        # Suppress linker warnings about files with no symbols defined.
        "-ignore:4221",
    ],
    "//conditions:default": ["-lpthread", "-lm"],
})

load(
    ":protobuf.bzl",
    "cc_proto_library",
    "py_proto_library",
    "internal_copied_filegroup",
)

cc_library(
    name = "protobuf_lite",
    srcs = [
        # AUTOGEN(protobuf_lite_srcs)
        "src/google/protobuf/any_lite.cc",
        "src/google/protobuf/arena.cc",
        "src/google/protobuf/extension_set.cc",
        "src/google/protobuf/generated_message_table_driven_lite.cc",
        "src/google/protobuf/generated_message_util.cc",
        "src/google/protobuf/implicit_weak_message.cc",
        "src/google/protobuf/io/coded_stream.cc",
        "src/google/protobuf/io/strtod.cc",
        "src/google/protobuf/io/io_win32.cc",
        "src/google/protobuf/io/coded_stream_ext.cc",
        "src/google/protobuf/io/zero_copy_stream_ext.cc",
        "src/google/protobuf/rawstring.cc",
        "src/google/protobuf/io/zero_copy_stream.cc",
        "src/google/protobuf/io/zero_copy_stream_impl_lite.cc",
        "src/google/protobuf/message_lite.cc",
        "src/google/protobuf/parse_context.cc",
        "src/google/protobuf/repeated_field.cc",
        "src/google/protobuf/stubs/bytestream.cc",
        "src/google/protobuf/stubs/common.cc",
        "src/google/protobuf/stubs/int128.cc",
        "src/google/protobuf/stubs/status.cc",
        "src/google/protobuf/stubs/statusor.cc",
        "src/google/protobuf/stubs/stringpiece.cc",
        "src/google/protobuf/stubs/stringprintf.cc",
        "src/google/protobuf/stubs/structurally_valid.cc",
        "src/google/protobuf/stubs/strutil.cc",
        "src/google/protobuf/stubs/time.cc",
        "src/google/protobuf/wire_format_lite.cc",
    ],
    hdrs = glob(["src/google/protobuf/**/*.h", "src/google/protobuf/**/*.inc"]),
    copts = COPTS,
    includes = ["src/"],
    linkopts = LINK_OPTS,
    visibility = ["//visibility:public"],
)

PROTOBUF_DEPS = select({
    ":msvc": [],
    "//conditions:default": ZLIB_DEPS,
})

cc_library(
    name = "protobuf",
    srcs = [
        # AUTOGEN(protobuf_srcs)
        "src/google/protobuf/any.cc",
        "src/google/protobuf/any.pb.cc",
        "src/google/protobuf/api.pb.cc",
        "src/google/protobuf/compiler/importer.cc",
        "src/google/protobuf/compiler/parser.cc",
        "src/google/protobuf/descriptor.cc",
        "src/google/protobuf/descriptor.pb.cc",
        "src/google/protobuf/descriptor_database.cc",
        "src/google/protobuf/duration.pb.cc",
        "src/google/protobuf/dynamic_message.cc",
        "src/google/protobuf/empty.pb.cc",
        "src/google/protobuf/extension_set_heavy.cc",
        "src/google/protobuf/field_mask.pb.cc",
        "src/google/protobuf/generated_message_reflection.cc",
        "src/google/protobuf/generated_message_table_driven.cc",
        "src/google/protobuf/io/gzip_stream.cc",
        "src/google/protobuf/io/printer.cc",
        "src/google/protobuf/io/tokenizer.cc",
        "src/google/protobuf/io/zero_copy_stream_impl.cc",
        "src/google/protobuf/map_field.cc",
        "src/google/protobuf/message.cc",
        "src/google/protobuf/reflection_ops.cc",
        "src/google/protobuf/service.cc",
        "src/google/protobuf/source_context.pb.cc",
        "src/google/protobuf/struct.pb.cc",
        "src/google/protobuf/stubs/mathlimits.cc",
        "src/google/protobuf/stubs/substitute.cc",
        "src/google/protobuf/text_format.cc",
        "src/google/protobuf/timestamp.pb.cc",
        "src/google/protobuf/type.pb.cc",
        "src/google/protobuf/unknown_field_set.cc",
        "src/google/protobuf/util/delimited_message_util.cc",
        "src/google/protobuf/util/field_comparator.cc",
        "src/google/protobuf/util/field_mask_util.cc",
        "src/google/protobuf/util/internal/datapiece.cc",
        "src/google/protobuf/util/internal/default_value_objectwriter.cc",
        "src/google/protobuf/util/internal/error_listener.cc",
        "src/google/protobuf/util/internal/field_mask_utility.cc",
        "src/google/protobuf/util/internal/json_escaping.cc",
        "src/google/protobuf/util/internal/json_objectwriter.cc",
        "src/google/protobuf/util/internal/json_stream_parser.cc",
        "src/google/protobuf/util/internal/object_writer.cc",
        "src/google/protobuf/util/internal/proto_writer.cc",
        "src/google/protobuf/util/internal/protostream_objectsource.cc",
        "src/google/protobuf/util/internal/protostream_objectwriter.cc",
        "src/google/protobuf/util/internal/type_info.cc",
        "src/google/protobuf/util/internal/type_info_test_helper.cc",
        "src/google/protobuf/util/internal/utility.cc",
        "src/google/protobuf/util/json_util.cc",
        "src/google/protobuf/util/message_differencer.cc",
        "src/google/protobuf/util/time_util.cc",
        "src/google/protobuf/util/type_resolver_util.cc",
        "src/google/protobuf/wire_format.cc",
        "src/google/protobuf/wrappers.pb.cc",
    ],
    hdrs = glob(["src/**/*.h", "src/**/*.inc"]),
    copts = COPTS,
    includes = ["src/"],
    linkopts = LINK_OPTS,
    visibility = ["//visibility:public"],
    deps = [":protobuf_lite"] + PROTOBUF_DEPS,
)

# This provides just the header files for use in projects that need to build
# shared libraries for dynamic loading. This target is available until Bazel
# adds native support for such use cases.
# TODO(keveman): Remove this target once the support gets added to Bazel.
cc_library(
    name = "protobuf_headers",
    hdrs = glob(["src/**/*.h"]),
    includes = ["src/"],
    visibility = ["//visibility:public"],
)

# Map of all well known protos.
# name => (include path, imports)
WELL_KNOWN_PROTO_MAP = {
    "any" : ("google/protobuf/any.proto", []),
    "api" : ("google/protobuf/api.proto", ["source_context", "type"]),
    "compiler_plugin" : ("google/protobuf/compiler/plugin.proto", ["descriptor"]),
    "descriptor" : ("google/protobuf/descriptor.proto", []),
    "duration" : ("google/protobuf/duration.proto", []),
    "empty" : ("google/protobuf/empty.proto", []),
    "field_mask" : ("google/protobuf/field_mask.proto", []),
    "source_context" : ("google/protobuf/source_context.proto", []),
    "struct" : ("google/protobuf/struct.proto", []),
    "timestamp" : ("google/protobuf/timestamp.proto", []),
    "type" : ("google/protobuf/type.proto", ["any", "source_context"]),
    "wrappers" : ("google/protobuf/wrappers.proto", []),
}

RELATIVE_WELL_KNOWN_PROTOS = [proto[1][0] for proto in WELL_KNOWN_PROTO_MAP.items()]

WELL_KNOWN_PROTOS = ["src/" + s for s in RELATIVE_WELL_KNOWN_PROTOS]

filegroup(
    name = "well_known_protos",
    srcs = WELL_KNOWN_PROTOS,
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "cc_wkt_protos",
    srcs = WELL_KNOWN_PROTOS,
    include = "src",
    default_runtime = ":protobuf",
    internal_bootstrap_hack = 1,
    protoc = ":protoc",
    visibility = ["//visibility:public"],
)

################################################################################
# Well Known Types Proto Library Rules
#
# These proto_library rules can be used with one of the language specific proto
# library rules i.e. java_proto_library:
#
# java_proto_library(
#   name = "any_java_proto",
#   deps = ["@com_google_protobuf//:any_proto],
# )
################################################################################

internal_copied_filegroup(
    name = "_internal_wkt_protos",
    srcs = WELL_KNOWN_PROTOS,
    dest = "",
    strip_prefix = "src",
    visibility = ["//visibility:private"],
)

[proto_library(
    name = proto[0] + "_proto",
    srcs = [proto[1][0]],
    deps = [dep + "_proto" for dep in proto[1][1]],
    visibility = ["//visibility:public"],
    ) for proto in WELL_KNOWN_PROTO_MAP.items()]

################################################################################
# Protocol Buffers Compiler
################################################################################

cc_library(
    name = "protoc_lib",
    srcs = [
        # AUTOGEN(protoc_lib_srcs)
        "src/google/protobuf/compiler/code_generator.cc",
        "src/google/protobuf/compiler/command_line_interface.cc",
        "src/google/protobuf/compiler/cpp/cpp_enum.cc",
        "src/google/protobuf/compiler/cpp/cpp_enum_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_extension.cc",
        "src/google/protobuf/compiler/cpp/cpp_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_file.cc",
        "src/google/protobuf/compiler/cpp/cpp_generator.cc",
        "src/google/protobuf/compiler/cpp/cpp_helpers.cc",
        "src/google/protobuf/compiler/cpp/cpp_map_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_message.cc",
        "src/google/protobuf/compiler/cpp/cpp_message_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_padding_optimizer.cc",
        "src/google/protobuf/compiler/cpp/cpp_primitive_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_service.cc",
        "src/google/protobuf/compiler/cpp/cpp_string_field.cc",
        "src/google/protobuf/compiler/cpp/cpp_rawstring_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_doc_comment.cc",
        "src/google/protobuf/compiler/csharp/csharp_enum.cc",
        "src/google/protobuf/compiler/csharp/csharp_enum_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_field_base.cc",
        "src/google/protobuf/compiler/csharp/csharp_generator.cc",
        "src/google/protobuf/compiler/csharp/csharp_helpers.cc",
        "src/google/protobuf/compiler/csharp/csharp_map_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_message.cc",
        "src/google/protobuf/compiler/csharp/csharp_message_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_primitive_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_reflection_class.cc",
        "src/google/protobuf/compiler/csharp/csharp_repeated_enum_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_repeated_message_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_repeated_primitive_field.cc",
        "src/google/protobuf/compiler/csharp/csharp_source_generator_base.cc",
        "src/google/protobuf/compiler/csharp/csharp_wrapper_field.cc",
        "src/google/protobuf/compiler/java/java_context.cc",
        "src/google/protobuf/compiler/java/java_doc_comment.cc",
        "src/google/protobuf/compiler/java/java_enum.cc",
        "src/google/protobuf/compiler/java/java_enum_field.cc",
        "src/google/protobuf/compiler/java/java_enum_field_lite.cc",
        "src/google/protobuf/compiler/java/java_enum_lite.cc",
        "src/google/protobuf/compiler/java/java_extension.cc",
        "src/google/protobuf/compiler/java/java_extension_lite.cc",
        "src/google/protobuf/compiler/java/java_field.cc",
        "src/google/protobuf/compiler/java/java_file.cc",
        "src/google/protobuf/compiler/java/java_generator.cc",
        "src/google/protobuf/compiler/java/java_generator_factory.cc",
        "src/google/protobuf/compiler/java/java_helpers.cc",
        "src/google/protobuf/compiler/java/java_map_field.cc",
        "src/google/protobuf/compiler/java/java_map_field_lite.cc",
        "src/google/protobuf/compiler/java/java_message.cc",
        "src/google/protobuf/compiler/java/java_message_builder.cc",
        "src/google/protobuf/compiler/java/java_message_builder_lite.cc",
        "src/google/protobuf/compiler/java/java_message_field.cc",
        "src/google/protobuf/compiler/java/java_message_field_lite.cc",
        "src/google/protobuf/compiler/java/java_message_lite.cc",
        "src/google/protobuf/compiler/java/java_name_resolver.cc",
        "src/google/protobuf/compiler/java/java_primitive_field.cc",
        "src/google/protobuf/compiler/java/java_primitive_field_lite.cc",
        "src/google/protobuf/compiler/java/java_service.cc",
        "src/google/protobuf/compiler/java/java_shared_code_generator.cc",
        "src/google/protobuf/compiler/java/java_string_field.cc",
        "src/google/protobuf/compiler/java/java_string_field_lite.cc",
        "src/google/protobuf/compiler/js/js_generator.cc",
        "src/google/protobuf/compiler/js/well_known_types_embed.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_enum.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_enum_field.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_extension.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_field.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_file.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_generator.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_helpers.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_map_field.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_message.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_message_field.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_oneof.cc",
        "src/google/protobuf/compiler/objectivec/objectivec_primitive_field.cc",
        "src/google/protobuf/compiler/php/php_generator.cc",
        "src/google/protobuf/compiler/plugin.cc",
        "src/google/protobuf/compiler/plugin.pb.cc",
        "src/google/protobuf/compiler/python/python_generator.cc",
        "src/google/protobuf/compiler/ruby/ruby_generator.cc",
        "src/google/protobuf/compiler/subprocess.cc",
        "src/google/protobuf/compiler/zip_writer.cc",
    ],
    copts = COPTS,
    includes = ["src/"],
    linkopts = LINK_OPTS + select({
        ":msvc": [
            # Linking to setargv.obj makes the default command line argument
            # parser expand wildcards, so the main method's argv will contain the
            # expanded list instead of the wildcards.
            #
            # Adding dummy "-DEFAULTLIB:kernel32.lib", because:
            # - Microsoft ships this object file next to default libraries
            # - but this file is not a library, just a precompiled object
            # - "-WHOLEARCHIVE" and "-DEFAULTLIB" only accept library,
            #   not precompiled object.
            # - Bazel would assume linkopt that does not start with "-" or "$"
            #   as a label to a target, so we add a harmless "-DEFAULTLIB:kernel32.lib"
            #   before "setargv.obj".
            # See https://msdn.microsoft.com/en-us/library/8bch7bkk.aspx
            "-DEFAULTLIB:kernel32.lib setargv.obj",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [":protobuf"],
)

cc_binary(
    name = "protoc",
    srcs = ["src/google/protobuf/compiler/main.cc"],
    linkopts = LINK_OPTS,
    visibility = ["//visibility:public"],
    deps = [":protoc_lib"],
)


py_library(
    name = "python_srcs",
    srcs = glob(
        [
            "python/google/__init__.py",
            "python/google/protobuf/*.py",
            "python/google/protobuf/**/*.py",
        ],
        exclude = [
            "python/google/protobuf/__init__.py",
            "python/google/protobuf/**/__init__.py",
        ],
    ),
    imports = ["python"],
    srcs_version = "PY2AND3",
)

cc_binary(
    name = "python/google/protobuf/internal/_api_implementation.so",
    srcs = ["python/google/protobuf/internal/api_implementation.cc"],
    copts = COPTS + [
        "-DPYTHON_PROTO2_CPP_IMPL_V2",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = select({
        "//conditions:default": [],
        ":use_fast_cpp_protos": ["//external:python_headers"],
    }),
)

cc_binary(
    name = "python/google/protobuf/pyext/_message.so",
    srcs = glob([
        "python/google/protobuf/pyext/*.cc",
        "python/google/protobuf/pyext/*.h",
    ]),
    copts = COPTS + [
        "-DGOOGLE_PROTOBUF_HAS_ONEOF=1",
    ] + select({
        "//conditions:default": [],
        ":allow_oversize_protos": ["-DPROTOBUF_PYTHON_ALLOW_OVERSIZE_PROTOS=1"],
    }),
    includes = [
        "python/",
        "src/",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":protobuf",
        ":proto_api",
    ] + select({
        "//conditions:default": [],
        ":use_fast_cpp_protos": ["//external:python_headers"],
    }),
)

config_setting(
    name = "use_fast_cpp_protos",
    values = {
        "define": "use_fast_cpp_protos=true",
    },
)

config_setting(
    name = "allow_oversize_protos",
    values = {
        "define": "allow_oversize_protos=true",
    },
)

# Copy the builtin proto files from src/google/protobuf to
# python/google/protobuf. This way, the generated Python sources will be in the
# same directory as the Python runtime sources. This is necessary for the
# modules to be imported correctly since they are all part of the same Python
# package.
internal_copied_filegroup(
    name = "protos_python",
    srcs = WELL_KNOWN_PROTOS,
    dest = "python",
    strip_prefix = "src",
)

# TODO(dzc): Remove this once py_proto_library can have labels in srcs, in
# which case we can simply add :protos_python in srcs.
COPIED_WELL_KNOWN_PROTOS = ["python/" + s for s in RELATIVE_WELL_KNOWN_PROTOS]

py_proto_library(
    name = "protobuf_python",
    srcs = COPIED_WELL_KNOWN_PROTOS,
    include = "python",
    data = select({
        "//conditions:default": [],
        ":use_fast_cpp_protos": [
            ":python/google/protobuf/internal/_api_implementation.so",
            ":python/google/protobuf/pyext/_message.so",
        ],
    }),
    default_runtime = "",
    protoc = ":protoc",
    py_libs = [
        ":python_srcs",
        "//external:six",
    ],
    py_extra_srcs = glob(["python/**/__init__.py"]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)


cc_library(
    name = "proto_api",
    hdrs = ["python/google/protobuf/proto_api.h"],
    deps = [
        "//external:python_headers",
    ],
    visibility = ["//visibility:public"],
)

proto_lang_toolchain(
    name = "cc_toolchain",
    command_line = "--cpp_out=$(OUT)",
    runtime = ":protobuf",
    visibility = ["//visibility:public"],
    blacklisted_protos = [":_internal_wkt_protos_genrule"],
)
