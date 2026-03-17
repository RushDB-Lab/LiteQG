cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(symphonyqg NB_STATIC bindings.cpp)

target_include_directories(symphonyqg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../symqglib
)

target_compile_definitions(symphonyqg PRIVATE
  VERSION_INFO=${PROJECT_VERSION}
  EIGEN_DONT_PARALLELIZE
)

target_compile_options(symphonyqg PRIVATE
  -Ofast
  -march=native
  -ftree-vectorize
  -fvisibility=hidden
  -Wall
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(symphonyqg PRIVATE OpenMP::OpenMP_CXX)
endif()

# -lrt is Linux-only
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(symphonyqg PRIVATE rt)
endif()

install(TARGETS symphonyqg LIBRARY DESTINATION .)
