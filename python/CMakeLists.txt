cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Fall back to C++17 if C++20 is not available
if(NOT CMAKE_CXX_STANDARD GREATER_EQUAL 17)
  message(FATAL_ERROR "At least C++17 support is required")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS NumPy REQUIRED)

pybind11_add_module(symphonyqg bindings.cpp)

target_include_directories(symphonyqg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../symqglib
  ${Python_NumPy_INCLUDE_DIRS}
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

# -lrt is Linux-only (not available on macOS)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(symphonyqg PRIVATE rt)
endif()

install(TARGETS symphonyqg LIBRARY DESTINATION .)
