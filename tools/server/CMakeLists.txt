include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
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

if (LLAMA_OTEL)
    # Follow the official opentelemetry-cpp getting started guide:
    # Build opentelemetry-cpp, install to a local prefix, then pass
    # -Dopentelemetry-cpp_DIR=<prefix>/lib/cmake/opentelemetry-cpp
    # opentelemetry-cpp-target.cmake references the CURL::libcurl imported
    # target from CMake's own FindCURL module without exporting it, so
    # the consumer must find_package(CURL) first.
    find_package(CURL REQUIRED)
    find_package(opentelemetry-cpp CONFIG REQUIRED)
    target_sources(${TARGET} PRIVATE server-otel.cpp server-otel.h)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_OTEL)
    # PUBLIC: server-otel.h is included from server.cpp (llama-server-impl),
    # which needs opentelemetry-cpp's include dirs to resolve it under
    # LLAMA_OTEL. PRIVATE would only satisfy server-context's own compile.
    target_link_libraries(${TARGET} PUBLIC ${OPENTELEMETRY_CPP_LIBRARIES})
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

if (LLAMA_OTEL)
    # server.cpp includes server-otel.h directly for the init/shutdown hooks;
    # the opentelemetry-cpp include dirs and libraries needed to resolve it
    # come transitively via the PUBLIC link to server-context above.
    target_compile_definitions(${TARGET} PRIVATE LLAMA_OTEL)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
