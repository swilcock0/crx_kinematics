#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model/joint_model_group.h>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include "crx_kinematics/crx_kinematics_plugin.hpp"
#include "crx_kinematics/robot.hpp"

namespace {
Eigen::Isometry3d from_xyzwpr(const std::array<double, 6>& xyzwpr)
{
  Eigen::Isometry3d T;
  T.translation().x() = xyzwpr[0] / 1000.0;
  T.translation().y() = xyzwpr[1] / 1000.0;
  T.translation().z() = xyzwpr[2] / 1000.0;
  Eigen::Quaterniond q =
      Eigen::AngleAxisd(xyzwpr[5] / 180.0 * M_PI, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(xyzwpr[4] / 180.0 * M_PI, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(xyzwpr[3] / 180.0 * M_PI, Eigen::Vector3d::UnitX());
  T.linear() = Eigen::Matrix3d(q);
  return T;
}
}

TEST(TestCrxKinematicsPlugin, returns_valid_solution_matching_pose)
{
  // Initialize ROS 2 rclcpp before creating any nodes
  int argc = 0;
  char** argv = nullptr;
  rclcpp::init(argc, argv);
  // Build a minimal 6-DOF robot model inline (base_link -> link6) and a group "manipulator"
  const char* urdf_str = R"(
  <robot name="test_arm">
    <link name="base_link"/>
    <joint name="J1" type="revolute">
      <parent link="base_link"/>
      <child link="link1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link1"/>
    <joint name="J2" type="revolute">
      <parent link="link1"/>
      <child link="link2"/>
      <origin xyz="0 0 0.71" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link2"/>
    <joint name="J3" type="revolute">
      <parent link="link2"/>
      <child link="link3"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link3"/>
    <joint name="J4" type="revolute">
      <parent link="link3"/>
      <child link="link4"/>
      <origin xyz="0 0 -0.54" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link4"/>
    <joint name="J5" type="revolute">
      <parent link="link4"/>
      <child link="link5"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link5"/>
    <joint name="J6" type="revolute">
      <parent link="link5"/>
      <child link="tool0"/>
      <origin xyz="0 0 -0.16" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="tool0"/>
  </robot>
  )";

  auto urdf = urdf::parseURDF(urdf_str);
  ASSERT_TRUE(urdf);

  const char* srdf_str = R"(
  <robot name="test_arm">
    <group name="manipulator">
      <chain base_link="base_link" tip_link="tool0"/>
    </group>
  </robot>
  )";
  srdf::ModelSharedPtr srdf(new srdf::Model);
  ASSERT_TRUE(srdf->initString(*urdf, srdf_str));

  moveit::core::RobotModelPtr robot_model(new moveit::core::RobotModel(urdf, srdf));
  ASSERT_TRUE(robot_model);

  // Create node
  auto node = rclcpp::Node::make_shared("crx_kinematics_plugin_test");

  // Initialize plugin
  crx_kinematics::CRXKinematicsPlugin plugin;
  const std::string group_name = "manipulator";
  const std::vector<std::string> tips = { "tool0" };
  ASSERT_TRUE(plugin.initialize(node, *robot_model, group_name, "base_link", tips, 0.0));

  // Build a target pose using CRXRobot FK from a nominal joint vector
  crx_kinematics::CRXRobot robot;
  std::array<double, 6> q_target{ 0.3, -0.5, 0.8, 0.2, -0.4, 0.6 };
  Eigen::Isometry3d target = robot.fk(q_target);
  geometry_msgs::msg::Pose pose_msg = tf2::toMsg(target);

  // Seed (size 6, values arbitrary)
  std::vector<double> seed(6, 0.0);

  std::vector<double> solution;
  moveit_msgs::msg::MoveItErrorCodes ec;
  kinematics::KinematicsQueryOptions opts;

  ASSERT_TRUE(plugin.getPositionIK(pose_msg, seed, solution, ec, opts));
  ASSERT_EQ(ec.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  ASSERT_EQ(solution.size(), 6u);

  // Verify FK matches target
  std::array<double, 6> q{};
  for (size_t i = 0; i < 6; ++i) q[i] = solution[i];
  Eigen::Isometry3d fk = robot.fk(q);
  EXPECT_TRUE(fk.translation().isApprox(target.translation(), 1e-6));
  EXPECT_TRUE(fk.linear().isApprox(target.linear(), 1e-6));

  // Shutdown ROS 2 after test completes
  rclcpp::shutdown();
}

namespace {
// Helper to initialize plugin with inline URDF/SRDF
std::unique_ptr<crx_kinematics::CRXKinematicsPlugin> make_plugin(rclcpp::Node::SharedPtr& node,
                                                                 moveit::core::RobotModelPtr& robot_model)
{
  const char* urdf_str = R"(
  <robot name="test_arm">
    <link name="base_link"/>
    <joint name="J1" type="revolute">
      <parent link="base_link"/>
      <child link="link1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link1"/>
    <joint name="J2" type="revolute">
      <parent link="link1"/>
      <child link="link2"/>
      <origin xyz="0 0 0.71" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link2"/>
    <joint name="J3" type="revolute">
      <parent link="link2"/>
      <child link="link3"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link3"/>
    <joint name="J4" type="revolute">
      <parent link="link3"/>
      <child link="link4"/>
      <origin xyz="0 0 -0.54" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link4"/>
    <joint name="J5" type="revolute">
      <parent link="link4"/>
      <child link="link5"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="link5"/>
    <joint name="J6" type="revolute">
      <parent link="link5"/>
      <child link="tool0"/>
      <origin xyz="0 0 -0.16" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit effort="100" velocity="1" lower="-3.14" upper="3.14"/>
    </joint>
    <link name="tool0"/>
  </robot>
  )";
  auto urdf = urdf::parseURDF(urdf_str);
  srdf::ModelSharedPtr srdf(new srdf::Model);
  const char* srdf_str = R"(
  <robot name="test_arm">
    <group name="manipulator">
      <chain base_link="base_link" tip_link="tool0"/>
    </group>
  </robot>
  )";
  srdf->initString(*urdf, srdf_str);
  robot_model.reset(new moveit::core::RobotModel(urdf, srdf));
  node = rclcpp::Node::make_shared("crx_kinematics_plugin_test_tables");
  auto plugin = std::make_unique<crx_kinematics::CRXKinematicsPlugin>();
  const std::string group_name = "manipulator";
  const std::vector<std::string> tips = { "tool0" };
  EXPECT_TRUE(plugin->initialize(node, *robot_model, group_name, "base_link", tips, 0.0));
  return plugin;
}

bool matches_expected(const std::vector<double>& sol_rad, const std::array<double, 6>& exp_deg)
{
  for (size_t i = 0; i < 6; ++i)
  {
    double deg = sol_rad[i] / M_PI * 180.0;
    if (std::abs(deg - exp_deg[i]) > 1e-1) return false;
  }
  return true;
}
}

TEST(TestCrxKinematicsPlugin, table_example_3_1)
{
  int argc = 0; char** argv = nullptr; rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node; moveit::core::RobotModelPtr robot_model;
  auto plugin = make_plugin(node, robot_model);

  Eigen::Isometry3d T = from_xyzwpr({ 80.321, 287.676, 394.356, -131.819, -45.268, 61.453 });
  geometry_msgs::msg::Pose pose = tf2::toMsg(T);

  std::vector<double> seed(6, 0.0), sol; moveit_msgs::msg::MoveItErrorCodes ec; kinematics::KinematicsQueryOptions opts;
  ASSERT_TRUE(plugin->getPositionIK(pose, seed, sol, ec, opts));
  ASSERT_EQ(ec.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);

  std::vector<std::array<double, 6>> expected = {
    { 44.611, 89.087, 109.193, 94.703, 121.416, 121.782 },
    { 35.162, 88.468, 140.150, -108.846, -111.920, -91.804 },
    { 29.462, -39.473, -8.392, 117.682, 85.679, -119.224 },
    { 78, -41, 17, -42, -60, 10 },
    { -135.389, -89.087, 70.807, -85.297, 121.416, 121.782 },
    { -144.839, -88.468, 39.850, 71.154, -111.920, -91.804 },
    { -150.538, 39.473, -171.608, -62.318, 85.679, -119.224 },
    { 102, 41, 163, 138, -60, 10 },
  };
  bool any = false; for (auto& e : expected) any = any || matches_expected(sol, e);
  EXPECT_TRUE(any);
  rclcpp::shutdown();
}

TEST(TestCrxKinematicsPlugin, table_example_3_2)
{
  int argc = 0; char** argv = nullptr; rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node; moveit::core::RobotModelPtr robot_model;
  auto plugin = make_plugin(node, robot_model);

  Eigen::Isometry3d T = from_xyzwpr({ 600, 0, 100, 180, 0, 70 });
  geometry_msgs::msg::Pose pose = tf2::toMsg(T);

  std::vector<double> seed(6, 0.0), sol; moveit_msgs::msg::MoveItErrorCodes ec; kinematics::KinematicsQueryOptions opts;
  ASSERT_TRUE(plugin->getPositionIK(pose, seed, sol, ec, opts));
  ASSERT_EQ(ec.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);

  std::vector<std::array<double, 6>> expected = {
    { -14.478, 119.780, 78.001, -180, 168.001, -95.522 },
    { -6.336, 121.233, 89.999, -116.197, 179.999, -39.861 },
    { 6.335, 121.233, 90, -63.811, 179.999, -0.146 },
    { 14.478, 119.780, 78.001, 0, -168.001, 55.522 },
    { -14.478, 11.999, -29.780, -180, 60.220, -95.522 },
    { 14.478, 11.999, -29.780, 0, -60.220, 55.522 },
    { 165.522, -119.780, 101.999, 0, 168.001, -95.522 },
    { 173.664, -121.233, 90.001, 63.811, 179.999, -39.861 },
    { -173.665, -121.233, 90, 116.190, 179.999, -0.146 },
    { -165.522, -119.780, 101.999, -180, -168.01, 55.522 },
    { 165.522, -11.999, -150.220, 0, 60.220, -95.522 },
    { -165.522, -11.999, -150.220, 180, -60.220, 55.522 },
  };
  bool any = false; for (auto& e : expected) any = any || matches_expected(sol, e);
  EXPECT_TRUE(any);
  rclcpp::shutdown();
}

TEST(TestCrxKinematicsPlugin, table_example_3_3)
{
  int argc = 0; char** argv = nullptr; rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node; moveit::core::RobotModelPtr robot_model;
  auto plugin = make_plugin(node, robot_model);

  Eigen::Isometry3d T = from_xyzwpr({ 209.470, -42.894, 685.496, -95.378, -64.226, -56.402 });
  geometry_msgs::msg::Pose pose = tf2::toMsg(T);

  std::vector<double> seed(6, 0.0), sol; moveit_msgs::msg::MoveItErrorCodes ec; kinematics::KinematicsQueryOptions opts;
  ASSERT_TRUE(plugin->getPositionIK(pose, seed, sol, ec, opts));
  ASSERT_EQ(ec.val, moveit_msgs::msg::MoveItErrorCodes::SUCCESS);

  std::vector<std::array<double, 6>> expected = {
    { -60.125, 62.707, 112.015, 90.165, 92.586, 132.291 },
    { -63.318, 62.684, 143.064, -93.111, -89.750, -78.691 },
    { 11.855, 54.151, 144.007, -28.773, -142.889, -48.616 },
    { 49.247, 46.825, 135.954, 29.379, -134.512, -5.010 },
    { -62.156, -40.094, 14.333, 92.039, 91.458, -129.964 },
    { 47.115, -53.924, 35.922, 28.105, -41.924, -48.121 },
    { 0, -45, 44, -37, -53, 0 },
    { -47.369, -40.310, 45.272, -78.410, -81.969, 17.952 },
    { 119.875, -62.707, 67.985, -89.835, 92.586, 132.291 },
    { 116.682, -62.684, 36.936, 86.889, -89.750, -78.691 },
    { -168.145, -54.151, 35.993, 151.227, -142.889, -48.616 },
    { -130.753, -46.825, 44.046, -150.621, -134.512, -5.01 },
    { 117.844, 40.094, 165.667, -87.961, 91.458, -129.964 },
    { -132.885, 53.924, 144.078, -151.895, -41.924, -48.121 },
    { -180, 45, 136, 143, -53, 0 },
    { 132.631, 40.310, 134.728, 101.590, -81.969, 17.952 },
  };
  bool any = false; for (auto& e : expected) any = any || matches_expected(sol, e);
  EXPECT_TRUE(any);
  rclcpp::shutdown();
}
