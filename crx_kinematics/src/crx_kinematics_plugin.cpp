#include "crx_kinematics/crx_kinematics_plugin.hpp"
#include "crx_kinematics/robot.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <moveit/robot_model/robot_model.h>

namespace crx_kinematics
{
// Static robot instance (one per plugin, shared across all instances)
static std::unique_ptr<CRXRobot> g_robot = nullptr;

bool CRXKinematicsPlugin::initialize(rclcpp::Node::SharedPtr const& node,
                                     moveit::core::RobotModel const& robot_model,
                                     std::string const& group_name,
                                     std::string const& base_frame,
                                     std::vector<std::string> const& tip_frames,
                                     double search_discretization)
{
    // Get the joint group from the robot model
    const moveit::core::JointModelGroup* jmg = robot_model.getJointModelGroup(group_name);
    if (!jmg)
        return false;

    // Store joint names from the joint model group
    joint_names_ = jmg->getJointModelNames();
    
    // Store link names (tip frames)
    link_names_ = tip_frames;

    // Initialize the robot if not already done
    if (!g_robot)
        g_robot = std::make_unique<CRXRobot>();

    return true;
}

bool CRXKinematicsPlugin::getPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                        const std::vector<double>& ik_seed_state,
                                        std::vector<double>& solution,
                                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                                        const kinematics::KinematicsQueryOptions& options) const
{
    if (!g_robot)
    {
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Convert geometry_msgs Pose to Eigen::Isometry3d
    Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();
    target_pose.translation() = Eigen::Vector3d(ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
    target_pose.linear() = Eigen::Quaterniond(ik_pose.orientation.w, ik_pose.orientation.x, 
                                               ik_pose.orientation.y, ik_pose.orientation.z).toRotationMatrix();

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Convert first solution from radians to degrees
    solution.clear();
    for (double angle : ik_solutions[0])
        solution.push_back(angle);

    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
}

bool CRXKinematicsPlugin::getPositionIK(const std::vector<geometry_msgs::msg::Pose>& ik_poses,
                                        const std::vector<double>& ik_seed_state,
                                        std::vector<std::vector<double>>& solutions,
                                        kinematics::KinematicsResult& result,
                                        const kinematics::KinematicsQueryOptions& options) const
{
    if (!g_robot || ik_poses.empty())
    {
        result.kinematic_error = kinematics::KinematicErrors::NO_SOLUTION;
        return false;
    }

    // For now, solve for the first pose only
    const auto& ik_pose = ik_poses[0];

    // Convert geometry_msgs Pose to Eigen::Isometry3d
    Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();
    target_pose.translation() = Eigen::Vector3d(ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
    target_pose.linear() = Eigen::Quaterniond(ik_pose.orientation.w, ik_pose.orientation.x, 
                                               ik_pose.orientation.y, ik_pose.orientation.z).toRotationMatrix();

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        result.kinematic_error = kinematics::KinematicErrors::NO_SOLUTION;
        return false;
    }

    // Convert all solutions to the output format
    solutions.clear();
    for (const auto& sol : ik_solutions)
    {
        std::vector<double> converted_sol;
        for (double angle : sol)
            converted_sol.push_back(angle);
        solutions.push_back(converted_sol);
    }

    result.kinematic_error = kinematics::KinematicErrors::OK;
    return true;
}

bool CRXKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state,
                                           double timeout,
                                           std::vector<double>& solution,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
    return getPositionIK(ik_pose, ik_seed_state, solution, error_code, options);
}

bool CRXKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state,
                                           double timeout,
                                           const std::vector<double>& consistency_limits,
                                           std::vector<double>& solution,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
    return getPositionIK(ik_pose, ik_seed_state, solution, error_code, options);
}

bool CRXKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state,
                                           double timeout,
                                           std::vector<double>& solution,
                                           const IKCallbackFn& solution_callback,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
    if (!g_robot)
    {
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Convert geometry_msgs Pose to Eigen::Isometry3d
    Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();
    target_pose.translation() = Eigen::Vector3d(ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
    target_pose.linear() = Eigen::Quaterniond(ik_pose.orientation.w, ik_pose.orientation.x, 
                                               ik_pose.orientation.y, ik_pose.orientation.z).toRotationMatrix();

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Call the callback for each solution
    for (const auto& sol : ik_solutions)
    {
        solution.clear();
        for (double angle : sol)
            solution.push_back(angle);

        error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
        solution_callback(ik_pose, solution, error_code);
    }

    return true;
}

bool CRXKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state,
                                           double timeout,
                                           const std::vector<double>& consistency_limits,
                                           std::vector<double>& solution,
                                           const IKCallbackFn& solution_callback,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
    return searchPositionIK(ik_pose, ik_seed_state, timeout, solution, solution_callback, error_code, options);
}

bool CRXKinematicsPlugin::getPositionFK(const std::vector<std::string>& link_names,
                                        const std::vector<double>& joint_angles,
                                        std::vector<geometry_msgs::msg::Pose>& poses) const
{
    if (!g_robot || joint_angles.size() != 6)
        return false;

    // Convert degrees to radians for the FK function
    std::array<double, 6> angles;
    std::copy(joint_angles.begin(), joint_angles.end(), angles.begin());

    // Compute the end-effector pose
    Eigen::Isometry3d T = g_robot->fk(angles);

    // Convert to geometry_msgs::Pose
    geometry_msgs::msg::Pose pose;
    pose.position.x = T.translation()[0];
    pose.position.y = T.translation()[1];
    pose.position.z = T.translation()[2];
    
    Eigen::Quaterniond q(T.linear());
    pose.orientation.w = q.w();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();

    poses.clear();
    for (const auto& link_name : link_names)
        poses.push_back(pose);

    return true;
}

std::vector<std::string> const& CRXKinematicsPlugin::getJointNames() const
{
    return joint_names_;
}

std::vector<std::string> const& CRXKinematicsPlugin::getLinkNames() const
{
    return link_names_;
}
}  // namespace crx_kinematics

PLUGINLIB_EXPORT_CLASS(crx_kinematics::CRXKinematicsPlugin, kinematics::KinematicsBase);
