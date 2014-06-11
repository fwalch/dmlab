cmake_minimum_required(VERSION 2.8.3)

project(FERdemo)

## Library dependencies
find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)

## Stasm
include(ExternalProject)

ExternalProject_Add(
  Stasm-Source
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/stasm
  URL http://www.milbo.org/stasm-files/2/stasm4.1.0.tar.gz
  URL_MD5 f5153c180b8f421e506ec671a30e50c5
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(Stasm-Source source_dir)
set(Stasm-Source_SOURCE_DIR ${source_dir})
message(${Stasm-Source_SOURCE_DIR})

ExternalProject_Add(
  Stasm
  GIT_REPOSITORY https://github.com/fwalch/stasm.git
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
  CMAKE_ARGS -DSTASM_BASE_DIR:STRING=${Stasm-Source_SOURCE_DIR}
  DEPENDS Stasm-Source
)
set(STASM_INCLUDE_DIRS ${Stasm-Source_SOURCE_DIR})
ExternalProject_Get_Property(Stasm binary_dir)
set(STASM_LIBRARIES ${binary_dir}/libstasm.a)

## Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -std=c++11")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/obj)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-duplicate-enum")
endif ()

## Sources
include_directories(${PYTHON_INCLUDE_DIRS} ${STASM_INCLUDE_DIRS})

add_executable(demo
  src/demo.cpp
)
add_dependencies(demo Stasm)

target_link_libraries(demo
  ${OpenCV_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${STASM_LIBRARIES}
)

