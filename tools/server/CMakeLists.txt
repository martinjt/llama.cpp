include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-chunk-cache.h
    server-chunk-cache-ram.cpp
    server-chunk-cache-disk.cpp
    server-chunk-cache-lmcache.cpp
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

option(LLAMA_OTEL "Enable OpenTelemetry trace export" OFF)

if (LLAMA_OTEL)
    # Follow the official opentelemetry-cpp getting started guide: build
    # opentelemetry-cpp, install to a local prefix, then pass
    # -Dopentelemetry-cpp_DIR=<prefix>/lib/cmake/opentelemetry-cpp.
    # opentelemetry-cpp-target.cmake references the CURL::libcurl imported
    # target from CMake's own FindCURL module without exporting it, so the
    # consumer must find_package(CURL) first.
    find_package(CURL REQUIRED)
    find_package(opentelemetry-cpp CONFIG REQUIRED)
    # server-otel lives in the server-context library because both
    # server-context.cpp (span start/end) and server.cpp (init/shutdown, in
    # llama-server-impl) call into it. The define/includes/libs are PUBLIC so
    # they propagate up to llama-server-impl and the executable, keeping the
    # LLAMA_OTEL-guarded otel_span layout in server-otel.h consistent across
    # every TU that includes it (avoids an ODR mismatch).
    target_sources(server-context PRIVATE server-otel.cpp server-otel.h)
    target_compile_definitions(server-context PUBLIC LLAMA_OTEL)
    target_link_libraries(server-context PUBLIC ${OPENTELEMETRY_CPP_LIBRARIES})
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
