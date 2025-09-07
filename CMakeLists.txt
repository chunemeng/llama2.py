cmake_minimum_required(VERSION 3.18)
project(MyCProject C)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置优化选项
if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-Ofast -march=native)
endif()

# 可执行文件

add_executable(run run.c win.c)
add_executable(runq runq.c win.c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")


## 链接数学库 (Linux/macOS)
#if(NOT MSVC)
#    target_link_libraries(run m)
#    target_link_libraries(runq m)
#endif()

# OpenMP 支持
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(run PUBLIC OpenMP::OpenMP_C)
    target_link_libraries(runq PUBLIC OpenMP::OpenMP_C)
endif()
