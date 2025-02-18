set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(indexing indexing.cpp)
add_executable(lazy_cleanup_set_benchmark lazy_cleanup_set_benchmark.cpp)