cmake_minimum_required(VERSION 3.24)
project(neural_network C)

set(CMAKE_C_STANDARD 11)

add_executable(neural_network main.c)

target_link_libraries(neural_network m)