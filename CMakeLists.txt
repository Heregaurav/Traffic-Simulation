cmake_minimum_required(VERSION 3.18)
project(LPSim LANGUAGES CXX CUDA)

# ---- CUDA Settings ----
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# Target architecture — change to match your GPU
# V100 = sm_70, A100 = sm_80, RTX 3090 = sm_86, RTX 4090 = sm_89
set(CMAKE_CUDA_ARCHITECTURES 70 80 86)

# ---- Compiler Flags ----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3")

# ---- Include Directories ----
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ---- Source Files ----
set(CUDA_SOURCES
    src/main.cu
    src/propagation.cu
    src/multi_gpu.cu
)
set(CXX_SOURCES
    src/partitioning.cpp
)

# ---- Executable ----
add_executable(lpsim ${CUDA_SOURCES} ${CXX_SOURCES})

# ---- CUDA Compilation Options ----
set_target_properties(lpsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---- Link Libraries ----
target_link_libraries(lpsim
    ${CMAKE_CUDA_LIBRARIES}
    cuda
    cudart
)

# ---- Enable Multi-GPU Peer Access ----
target_compile_definitions(lpsim PRIVATE ENABLE_P2P=1)

# ---- Installation ----
install(TARGETS lpsim DESTINATION bin)

message(STATUS "LPSim configured for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
