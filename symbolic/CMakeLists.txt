cmake_minimum_required(VERSION 3.22)
project(test C)

set(CMAKE_C_STANDARD 11)

add_executable(main main_new.c matrix.c nnet.c nnet.h matrix.h interval.h)
