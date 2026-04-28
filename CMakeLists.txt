cmake_minimum_required(VERSION 3.10)
project(hik_camera LANGUAGES CXX)

# ==================== 编译选项 ====================
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_EXPORT_COMPILE_COMMANDS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ==================== 查找依赖库 ====================
# 1. OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# 2. 海康 MVS SDK（手动指定路径）
set(MVS_SDK_PATH "/opt/MVS" CACHE PATH "Path to Hikvision MVS SDK")

find_path(MVS_INCLUDE_DIR 
    NAMES MvCameraControl.h 
    PATHS ${MVS_SDK_PATH}/include 
          /usr/local/include
    NO_DEFAULT_PATH
)

find_library(MVS_LIBRARY 
    NAMES MvCameraControl 
    PATHS ${MVS_SDK_PATH}/lib/64 
          ${MVS_SDK_PATH}/lib 
          ${MVS_SDK_PATH}/lib64 
          /usr/local/lib
    NO_DEFAULT_PATH
)

if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
    message(STATUS "✓ Found Hikvision MVS SDK")
    message(STATUS "  Include: ${MVS_INCLUDE_DIR}")
    message(STATUS "  Library: ${MVS_LIBRARY}")
else()
    message(WARNING "✗ Hikvision MVS SDK not found!")
    message(WARNING "  Please check MVS_SDK_PATH: ${MVS_SDK_PATH}")
    message(WARNING "  Expected structure:")
    message(WARNING "    ${MVS_SDK_PATH}/include/MvCameraControl.h")
    message(WARNING "    ${MVS_SDK_PATH}/lib/64/libMvCameraControl.so")
endif()

# ==================== 线程库 ====================
find_package(Threads REQUIRED)

# ==================== 源文件列表 ====================
set(SOURCES
    src/hik.cpp
    src/threadpool.cpp
)

# ==================== 可执行目标 ====================
add_executable(hik_camera_demo ${SOURCES})

# 包含目录
target_include_directories(hik_camera_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(hik_camera_demo PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
    target_include_directories(hik_camera_demo PRIVATE ${MVS_INCLUDE_DIR})
    target_link_libraries(hik_camera_demo PRIVATE ${MVS_LIBRARY})
    
    # 设置 RPATH，方便运行时查找动态库
    set_target_properties(hik_camera_demo PROPERTIES
        INSTALL_RPATH "${MVS_SDK_PATH}/lib/64"
        BUILD_RPATH "${MVS_SDK_PATH}/lib/64"
    )
endif()

# ==================== 输出信息 ====================
message(STATUS "")
message(STATUS "========== Project Configuration ==========")
message(STATUS "Project name:     ${PROJECT_NAME}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "Source files:     ${SOURCES}")
message(STATUS "OpenCV:           ${OpenCV_VERSION}")
if(MVS_INCLUDE_DIR AND MVS_LIBRARY)
    message(STATUS "Hikvision SDK:    Found")
else()
    message(STATUS "Hikvision SDK:    Not Found")
endif()
message(STATUS "==========================================")
message(STATUS "")