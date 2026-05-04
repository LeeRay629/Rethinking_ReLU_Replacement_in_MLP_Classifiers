cmake_minimum_required(VERSION 3.10)
project(FHE_CKKS VERSION 1.0)

# CMAKE config
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Debugging / build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(SEAL 4.1 REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    run_he_inference_otto
    ${CMAKE_SOURCE_DIR}/src/run_he_inference_otto.cpp
)

target_link_libraries(run_he_inference_otto PRIVATE
    SEAL::seal
    Threads::Threads
)

# Performance flags (CKKS is heavy on FFT/NTT; -O3 + native CPU helps a lot)
target_compile_options(run_he_inference_otto PRIVATE
    -O3
    -march=native
    -Wall
)



add_executable(
    run_he_inference_fgvcaircraft
    ${CMAKE_SOURCE_DIR}/src/run_he_inference_fgvcaircraft.cpp
)

target_link_libraries(run_he_inference_fgvcaircraft PRIVATE
    SEAL::seal
    Threads::Threads
)

# Performance flags (CKKS is heavy on FFT/NTT; -O3 + native CPU helps a lot)
target_compile_options(run_he_inference_fgvcaircraft PRIVATE
    -O3
    -march=native
    -Wall
)




add_executable(
    run_he_inference_massive
    ${CMAKE_SOURCE_DIR}/src/run_he_inference_massive.cpp
)

target_link_libraries(run_he_inference_massive PRIVATE
    SEAL::seal
    Threads::Threads
)

# Performance flags (CKKS is heavy on FFT/NTT; -O3 + native CPU helps a lot)
target_compile_options(run_he_inference_massive PRIVATE
    -O3
    -march=native
    -Wall
)