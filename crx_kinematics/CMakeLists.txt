cmake_minimum_required(VERSION 3.20)
project(crx_kinematics)
set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(moveit_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_kdl REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# Try to locate the MoveIt test utils library directly
find_library(MOVEIT_UTILS_LIB
  NAMES moveit_utils
  HINTS /opt/ros/$ENV{ROS_DISTRO}/lib /opt/ros/humble/lib
)

add_library(robot STATIC src/robot.cpp)
target_include_directories(
  robot PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(robot PUBLIC Eigen3::Eigen)
target_compile_options(robot PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(crx_kinematics_plugin SHARED src/crx_kinematics_plugin.cpp)
target_include_directories(
  crx_kinematics_plugin PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                               $<INSTALL_INTERFACE:include/crx_kinematics>)
target_link_libraries(
  crx_kinematics_plugin
  PUBLIC moveit_core::moveit_robot_model
  PRIVATE robot moveit_core::moveit_kinematics_base
          moveit_core::moveit_robot_state pluginlib::pluginlib rclcpp::rclcpp
          tf2_eigen::tf2_eigen tf2_kdl::tf2_kdl tf2_ros::tf2_ros tf2_geometry_msgs::tf2_geometry_msgs)

install(DIRECTORY include/ DESTINATION include/crx_kinematics)
install(
  TARGETS crx_kinematics_plugin
  EXPORT export_crx_kinematics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
pluginlib_export_plugin_description_file(moveit_core
                                         crx_kinematics_plugin_description.xml)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_fk_ik test/test_fk_ik.cpp)
  target_include_directories(test_fk_ik PRIVATE include)
  target_link_libraries(test_fk_ik robot)

  ament_add_gtest(test_plugin test/test_plugin.cpp)
  target_include_directories(test_plugin PRIVATE include)
  # If the raw shared library was found, create an imported target and link it.
  if(MOVEIT_UTILS_LIB)
    add_library(moveit_utils_imported UNKNOWN IMPORTED)
    set_target_properties(moveit_utils_imported PROPERTIES IMPORTED_LOCATION "${MOVEIT_UTILS_LIB}")
  else()
    message(WARNING "moveit_utils library not found; attempting to link without it. If undefined references to loadTestingRobotModel occur, set MOVEIT_UTILS_LIB.")
  endif()
  # Consolidate all links into a single plain-signature call to match ament's usage
  target_link_libraries(test_plugin
    $<$<BOOL:${MOVEIT_UTILS_LIB}>:moveit_utils_imported>
    robot
    crx_kinematics_plugin
    moveit_core::moveit_robot_model
    tf2_eigen::tf2_eigen
    rclcpp::rclcpp)
endif()

ament_export_targets(export_crx_kinematics HAS_LIBRARY_TARGET)
ament_export_dependencies(moveit_core pluginlibv3 rclcpp)
ament_package()
