
py_library(
    name = "kmon_flume",
    srcs = glob(["python_client/flume/*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:thrift"],
)

py_library(
    name = "kmonitor_py",
    srcs = glob([
        "python_client/kmonitor/*.py",
        "python_client/kmonitor/**/*.py",
    ]),
    imports = ["./python_client/"],
    visibility = ["//visibility:public"],
    deps = [":kmon_flume"],
)
