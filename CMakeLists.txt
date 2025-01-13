cmake_minimum_required(VERSION 3.18)

project(disco VERSION 0.1.1 LANGUAGES CXX)

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
