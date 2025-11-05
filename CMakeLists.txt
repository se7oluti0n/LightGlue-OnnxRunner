cmake_minimum_required(VERSION 3.0)
project(LightGlue-OnnxRunner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


message(STATUS "Project: LightGlue-OnnxRunner")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

# include(cmake/platform.cmake)  # checking platform

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
set(ONNXRUNTIME_GPU_DIR  /home/dell/sources/onnxruntime-linux-x64-gpu-1.18.0)
set(ONNXRUNTIME_DIR  /home/dell/sources/onnxruntime-linux-x64-1.18.0)

# message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")
# message(STATUS "ONNXRUNTIME_GPU_DIR Path: ${ONNXRUNTIME_GPU_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIB onnxruntime onnxruntime_providers_shared)

include_directories(${ONNXRUNTIME_GPU_DIR}/include)
link_directories("${ONNXRUNTIME_GPU_DIR}/lib")
set(ONNXRUNTIME_GPU_LIB onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)

# set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv480/build)
# message(STATUS "OPENCV_DIR Path: ${OPENCV_DIR}")
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${OPENCV_DIR}/include)
# link_directories("${OPENCV_DIR}/x64/vc16/lib")
# set(OpenCV_LIB opencv_world480)
find_package(OpenCV 4.12.0 REQUIRED)


# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)
add_executable(main ${SRC_LIST})
target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main ${OpenCV_LIB} ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
target_link_libraries(main ${OpenCV_LIB} ${ONNXRUNTIME_GPU_LIB} ${OpenCV_LIBS})

