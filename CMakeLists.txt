cmake_minimum_required(VERSION 3.18)

project(disco VERSION 0.1.3 LANGUAGES CXX)

include(GNUInstallDirs)

add_library(disco INTERFACE)
add_library(disco::disco ALIAS disco)

target_compile_features(disco INTERFACE cxx_std_20)

target_include_directories(
    disco
    INTERFACE
    "${PROJECT_SOURCE_DIR}/include/"
)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(BUILD_TESTING "" ON)
    option(BUILD_EXAMPLE "" ON)

    if(BUILD_TESTING)
        add_executable(test test/main.cpp)
        target_link_libraries(test PRIVATE disco::disco)
        if(NOT MSVC)
            target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
    endif()

    if(BUILD_EXAMPLE)
        add_executable(movielens examples/movielens.cpp)
        target_link_libraries(movielens PRIVATE disco::disco)
    endif()
endif()
