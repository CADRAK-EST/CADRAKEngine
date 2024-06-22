cmake_minimum_required(VERSION 3.1)
project(cgal_alpha_shape)

set(CMAKE_CXX_STANDARD 14)

find_package(CGAL REQUIRED COMPONENTS Core)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

add_library(cgal_alpha_shape MODULE alpha_shape.cpp)
target_link_libraries(cgal_alpha_shape PRIVATE CGAL::CGAL CGAL::CGAL_Core pybind11::module)