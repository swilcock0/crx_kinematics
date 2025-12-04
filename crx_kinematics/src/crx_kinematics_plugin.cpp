#include "crx_kinematics/crx_kinematics_plugin.hpp"
#include "crx_kinematics/robot.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <tf2_kdl/tf2_kdl.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Geometry>
// TODO: Remove conditional include when released to all active distros.
#if __has_include(<tf2/transform_datatypes.hpp>)
#include <tf2/transform_datatypes.hpp>
#else
#include <tf2/transform_datatypes.h>
#endif

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
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                "Initializing CRXKinematicsPlugin for group: %s", group_name.c_str());
    
    // Call storeValues from base class to initialize protected members
    // This is essential - the base class stores robot_model_, group_name_, base_frame_, tip_frames_
    storeValues(robot_model, group_name, base_frame, tip_frames, search_discretization);
    
    // Get the joint group from the robot model
    const moveit::core::JointModelGroup* jmg = robot_model.getJointModelGroup(group_name);
    if (!jmg)
    {
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                     "Could not find joint model group: %s", group_name.c_str());
        return false;
    }

    // Store joint names from the joint model group
    joint_names_ = jmg->getJointModelNames();
    
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                "Found %lu joints", joint_names_.size());
    for (const auto& name : joint_names_)
        RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER, "  - %s", name.c_str());
    
    // Store link names - should include at least the tip frames
    // For now, just store the tip frames (end-effector links)
    if (tip_frames.empty())
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER,
                    "No tip frames provided for group: %s - plugin may not work correctly", group_name.c_str());
        // Don't fail, just log warning - tip frames might be added later
    }
    else
    {
        link_names_ = tip_frames;
        
        RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                    "Configured %lu tip frames", link_names_.size());
        for (const auto& name : link_names_)
            RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "  - %s", name.c_str());
    }

    // Initialize the robot if not already done
    if (!g_robot)
    {
        g_robot = std::make_unique<CRXRobot>();
        RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "CRXRobot instance created");
    }

    // Set planning frame from parameter, default to base_frame
    planning_frame_ = base_frame;
    // Try to read optional planning_frame parameter; do not throw if undeclared
    if (node->has_parameter("planning_frame"))
    {
        node->get_parameter("planning_frame", planning_frame_);
    }
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "CRX plugin base_frame=%s planning_frame=%s",
                base_frame.c_str(), planning_frame_.c_str());
    if (planning_frame_ != base_frame)
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

    // Compute fixed offset from tool0 to actual tip (if different)
    tip_offset_ = Eigen::Isometry3d::Identity();
    if (!link_names_.empty())
    {
        const std::string tip = link_names_.front();
        if (tip != "flange")
        {
            moveit::core::RobotState rs(robot_model_);
            rs.setToDefaultValues();
            Eigen::Isometry3d T_base_tool0 = rs.getGlobalLinkTransform("flange");
            Eigen::Isometry3d T_base_tip = rs.getGlobalLinkTransform(tip);
            tip_offset_ = T_base_tool0.inverse() * T_base_tip;
            RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "Computed tip offset flange->%s", tip.c_str());
        }
    }

    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "CRXKinematicsPlugin initialized successfully");
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
        RCLCPP_ERROR(rclcpp::get_logger("crx_kinematics"), "Robot instance not initialized");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // If planning frame differs, transform pose into base_frame_
    geometry_msgs::msg::Pose pose_in_base = ik_pose;
    if (!planning_frame_.empty() && planning_frame_ != base_frame_ && tf_buffer_)
    {
        try
        {
            geometry_msgs::msg::TransformStamped tf = tf_buffer_->lookupTransform(base_frame_, planning_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2));
            tf2::doTransform(ik_pose, pose_in_base, tf);
            RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Transformed pose from %s to %s", planning_frame_.c_str(), base_frame_.c_str());
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_WARN(kinematics::KinematicsBase::LOGGER, "TF transform %s->%s failed: %s", planning_frame_.c_str(), base_frame_.c_str(), ex.what());
        }
    }
    // Convert geometry_msgs Pose to Eigen::Isometry3d using tf2_eigen
    Eigen::Isometry3d target_pose;
    tf2::fromMsg(pose_in_base, target_pose);

    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "IK input pose: pos=[%.4f, %.4f, %.4f]", ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Pose in base: pos=[%.4f, %.4f, %.4f]", pose_in_base.position.x, pose_in_base.position.y, pose_in_base.position.z);

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Computing IK for pose: [%.3f, %.3f, %.3f]", 
                 ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions (radians)
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("crx_kinematics"),
                    "No IK solutions found for pose [%.3f, %.3f, %.3f]",
                    ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Prepare joint limits from the RobotModel (similar to KDL plugin approach)
    const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(getGroupName());
    std::vector<std::pair<double,double>> limits;
    limits.reserve(6);
    if (jmg)
    {
        const auto& joint_models = jmg->getJointModels();
        for (const auto* jm : joint_models)
        {
            const auto& bounds = jm->getVariableBoundsMsg();
            for (const auto& b : bounds)
            {
                limits.emplace_back(b.min_position, b.max_position);
            }
        }
    }

    auto within_limits = [&limits](const std::array<double,6>& q){
        if (limits.size() != 6) return true; // fallback if limits missing
        for (size_t i=0;i<6;++i)
        {
            if (q[i] < limits[i].first - 1e-6 || q[i] > limits[i].second + 1e-6)
                return false;
        }
        return true;
    };

    // Select best solution
    double best_cost = std::numeric_limits<double>::infinity();
    std::array<double,6> best{};
    for (const auto& sol : ik_solutions)
    {
        if (!within_limits(sol))
            continue;
        // Compute FK to tool0 then to actual tip
        Eigen::Isometry3d T_tool = g_robot->fk(sol);
        Eigen::Isometry3d T = T_tool * tip_offset_;
        const Eigen::Vector3d dp = T.translation() - target_pose.translation();
        const Eigen::Matrix3d R_err = T.linear().transpose() * target_pose.linear();
        const double angle = std::acos(std::min(1.0, std::max(-1.0, (R_err.trace() - 1.0) / 2.0)));
        double cost = dp.squaredNorm() + 0.5 * angle * angle;
        // If a non-empty seed is provided (RViz passes current state), bias towards it
        if (ik_seed_state.size() == 6)
        {
            double seed_cost = 0.0;
            for (size_t i = 0; i < 6; ++i)
            {
                const double d = sol[i] - ik_seed_state[i];
                seed_cost += d * d;
            }
            // Small weight to prefer the same branch without overriding pose accuracy
            cost += 0.05 * seed_cost;
        }
        if (cost < best_cost)
        {
            best_cost = cost;
            best = sol;
        }
    }

    if (!std::isfinite(best_cost))
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER, "No valid IK solution matched target pose");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }
    // Validate best solution against strict thresholds and log diagnostics
    Eigen::Isometry3d T_best = g_robot->fk(best) * tip_offset_;
    const Eigen::Vector3d dp_best = T_best.translation() - target_pose.translation();
    const Eigen::Matrix3d R_err_best = T_best.linear().transpose() * target_pose.linear();
    const double angle_best = std::acos(std::min(1.0, std::max(-1.0, (R_err_best.trace() - 1.0) / 2.0)));
    const Eigen::Quaterniond q_target(target_pose.linear());
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                "Target pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                target_pose.translation().x(), target_pose.translation().y(), target_pose.translation().z(),
                q_target.w(), q_target.x(), q_target.y(), q_target.z());
    const Eigen::Quaterniond q_best(T_best.linear());
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                "Best   pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f], dpos=%.6f, ang_err=%.6f",
                T_best.translation().x(), T_best.translation().y(), T_best.translation().z(),
                q_best.w(), q_best.x(), q_best.y(), q_best.z(), dp_best.norm(), angle_best);
    if (dp_best.norm() > 1e-3 || angle_best > 1e-2)
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER,
                    "Best IK does not meet thresholds (dpos=%.6f, ang=%.6f)", dp_best.norm(), angle_best);
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Map solution to group variable order explicitly
    const moveit::core::JointModelGroup* jmg2 = robot_model_->getJointModelGroup(getGroupName());
    if (jmg2)
    {
        const auto& var_names = jmg2->getVariableNames();
        solution.resize(var_names.size());
        for (size_t i = 0; i < var_names.size(); ++i)
        {
            // Assuming joints named J1..J6 map to indices 0..5
            int idx = -1;
            if (var_names[i] == "J1") idx = 0;
            else if (var_names[i] == "J2") idx = 1;
            else if (var_names[i] == "J3") idx = 2;
            else if (var_names[i] == "J4") idx = 3;
            else if (var_names[i] == "J5") idx = 4;
            else if (var_names[i] == "J6") idx = 5;
            if (idx >= 0)
                solution[i] = best[idx];
            else
                solution[i] = best[i < 6 ? i : 0];
        }
    }
    else
    {
        solution.assign(best.begin(), best.end());
    }
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
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                     "Robot not initialized or no poses provided");
        result.kinematic_error = kinematics::KinematicErrors::NO_SOLUTION;
        return false;
    }

    // For now, solve for the first pose only
    const auto& ik_pose = ik_poses[0];

    // Convert geometry_msgs Pose to Eigen::Isometry3d using tf2_eigen
    Eigen::Isometry3d target_pose;
    tf2::fromMsg(ik_pose, target_pose);

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Computing multi-solution IK for pose: [%.3f, %.3f, %.3f]",
                 ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER,
                    "No IK solutions found for pose [%.3f, %.3f, %.3f]",
                    ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);
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

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Found %lu IK solutions",
                 solutions.size());

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
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER, "Robot instance not initialized");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Convert geometry_msgs Pose to Eigen::Isometry3d using tf2_eigen
    Eigen::Isometry3d target_pose;
    tf2::fromMsg(ik_pose, target_pose);

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Searching for IK solutions (timeout: %.3f) for pose: [%.3f, %.3f, %.3f]",
                 timeout, ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(target_pose);

    if (ik_solutions.empty())
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER,
                    "No IK solutions found");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Call the callback for each solution
    for (size_t i = 0; i < ik_solutions.size(); ++i)
    {
        solution.clear();
        for (double angle : ik_solutions[i])
            solution.push_back(angle);

        error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
        solution_callback(ik_pose, solution, error_code);
        
        RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                     "Called solution callback for solution %lu/%lu",
                     i + 1, ik_solutions.size());
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
    if (!g_robot)
    {
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER, "Robot instance not initialized");
        return false;
    }

    if (joint_angles.size() != 6)
    {
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                     "Expected 6 joint angles, got %lu", joint_angles.size());
        return false;
    }

    // Convert degrees to radians for the FK function
    std::array<double, 6> angles;
    std::copy(joint_angles.begin(), joint_angles.end(), angles.begin());

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Computing FK for joint angles: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                 joint_angles[0], joint_angles[1], joint_angles[2],
                 joint_angles[3], joint_angles[4], joint_angles[5]);

    // Compute the end-effector pose
    Eigen::Isometry3d T = g_robot->fk(angles);

    // Convert to geometry_msgs::Pose using tf2_eigen
    geometry_msgs::msg::Pose pose;
    pose = tf2::toMsg(T);

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "FK result: pos=[%.3f, %.3f, %.3f], quat=[%.3f, %.3f, %.3f, %.3f]",
                 pose.position.x, pose.position.y, pose.position.z,
                 pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);

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
