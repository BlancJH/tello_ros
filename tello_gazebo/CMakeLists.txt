cmake_minimum_required(VERSION 3.16)
project(tello_gazebo)

#=============
# Setup
#=============

# Default to C99
if (NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif ()

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif ()

# Emulate colcon
if ($ENV{CLION_IDE})
  message("Running inside CLion")
  set(tello_msgs_DIR "${PROJECT_SOURCE_DIR}/../../../install/tello_msgs/share/tello_msgs/cmake")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRUN_INSIDE_CLION")
endif ()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tello_msgs REQUIRED)
et(GZ_SIM_VER 7)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim${GZ_SIM_VER} REQUIRED)
find_package(gz-math7 REQUIRED)

#=============
# Tello Gazebo Harmonic plugin
#=============

add_library(
  TelloPlugin SHARED
  src/tello_plugin.cpp
)

target_include_directories(TelloPlugin PUBLIC include)

ament_target_dependencies(
  TelloPlugin
  geometry_msgs
  rclcpp
  std_msgs
  tello_msgs
)

target_link_libraries(TelloPlugin
  PUBLIC
    gz-plugin2::register
    gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
    gz-math7::gz-math7
)

#=============
# Install
#=============

# Install targets
install(
  TARGETS TelloPlugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

## Install world and launch files
install(
  DIRECTORY models worlds launch
  DESTINATION share/${PROJECT_NAME}
)

## Install Python scripts
install(
  PROGRAMS src/inject_entity.py
  DESTINATION lib/${PROJECT_NAME}
)

#=============
# Tests
#=============

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

#=============
# Run ament macros
#=============

ament_package()
