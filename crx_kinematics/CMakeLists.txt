cmake_minimum_required(VERSION 3.20)
project(crx_kinematics)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra -Werror -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(robot src/robot.cpp)
target_include_directories(
  robot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(robot PUBLIC Eigen3::Eigen)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_fk_ik test/test_fk_ik.cpp)
  target_include_directories(test_fk_ik PRIVATE include)
  target_link_libraries(test_fk_ik robot)
endif()

ament_package()
