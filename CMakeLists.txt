cmake_minimum_required(VERSION 3.0.2)
project(exp_final)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(exp_final src/exp_final.cpp)

target_link_libraries(exp_final
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

