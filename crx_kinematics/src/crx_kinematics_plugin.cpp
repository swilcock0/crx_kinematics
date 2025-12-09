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

    // Determine the planning frame. Default to 'cylinder_stand' for now.
    std::string planning_frame = "cylinder_stand";
    rclcpp::Parameter planning_frame_param;
    if (node->get_parameter("planning_frame", planning_frame_param))
    {
        planning_frame = planning_frame_param.as_string();
    }
    // Print planning frame
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                "Using planning frame: %s", planning_frame.c_str());

    // Store values, using the planning_frame as the base for the solver.
    // The 'base_frame' from MoveIt is the robot's root link.
    storeValues(robot_model, group_name, planning_frame, tip_frames, search_discretization);
    
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

    // The plugin's base_frame_ is now the planning_frame. We need to transform
    // from this frame to the robot's actual base frame (e.g., 'base_link').
    robot_base_frame_ = base_frame;
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "CRX plugin solver base_frame=%s, robot base_frame=%s",
                base_frame_.c_str(), robot_base_frame_.c_str());

    if (base_frame_ != robot_base_frame_)
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

    // Subscribe to the interactive marker topic for debugging
    marker_subscriber_ = node->create_subscription<visualization_msgs::msg::InteractiveMarkerUpdate>(
        "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/update",
        10,
        [this](const visualization_msgs::msg::InteractiveMarkerUpdate::SharedPtr msg) {
            markerUpdateCallback(msg);
        });

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
            Eigen::Vector3d trans = tip_offset_.translation();
            Eigen::Quaterniond rot(tip_offset_.linear());
            RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "tip_offset trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
                        trans.x(), trans.y(), trans.z(), rot.w(), rot.x(), rot.y(), rot.z());
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

    // The ik_pose is in the plugin's base_frame_ (planning_frame).
    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"),
                "IK request received in frame '%s': pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                base_frame_.c_str(),
                ik_pose.position.x, ik_pose.position.y, ik_pose.position.z,
                ik_pose.orientation.w, ik_pose.orientation.x, ik_pose.orientation.y, ik_pose.orientation.z);

    // Log the latest marker pose for comparison
    {
        std::lock_guard<std::mutex> lock(marker_pose_mutex_);
        RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"),
                    "Latest marker pose in frame '%s': pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                    "cylinder_stand", // Assuming this frame, as seen in the topic output
                    latest_marker_pose_.position.x, latest_marker_pose_.position.y, latest_marker_pose_.position.z,
                    latest_marker_pose_.orientation.w, latest_marker_pose_.orientation.x, latest_marker_pose_.orientation.y, latest_marker_pose_.orientation.z);
    }

    // Transform it to the robot's base frame for the IK solver.
    geometry_msgs::msg::Pose pose_in_robot_base = ik_pose;
    if (base_frame_ != robot_base_frame_ && tf_buffer_)
    {
        try
        {
            geometry_msgs::msg::TransformStamped tf = tf_buffer_->lookupTransform(robot_base_frame_, base_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2));
            tf2::doTransform(ik_pose, pose_in_robot_base, tf);
            RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Transformed pose from %s to %s", base_frame_.c_str(), robot_base_frame_.c_str());
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_WARN(kinematics::KinematicsBase::LOGGER, "TF transform %s->%s failed: %s", base_frame_.c_str(), robot_base_frame_.c_str(), ex.what());
            error_code.val = moveit_msgs::msg::MoveItErrorCodes::FRAME_TRANSFORM_FAILURE;
            return false;
        }
    }
    // Convert geometry_msgs Pose to Eigen::Isometry3d using tf2_eigen
    Eigen::Isometry3d target_pose;
    tf2::fromMsg(pose_in_robot_base, target_pose);

    // Adjust for tip offset: IK expects pose of flange, but target is for tip
    Eigen::Isometry3d desired_flange = target_pose * tip_offset_.inverse();

    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "IK input pose: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                ik_pose.position.x, ik_pose.position.y, ik_pose.position.z,
                ik_pose.orientation.w, ik_pose.orientation.x, ik_pose.orientation.y, ik_pose.orientation.z);
    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Pose in base: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                pose_in_robot_base.position.x, pose_in_robot_base.position.y, pose_in_robot_base.position.z,
                pose_in_robot_base.orientation.w, pose_in_robot_base.orientation.x, pose_in_robot_base.orientation.y, pose_in_robot_base.orientation.z);

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Computing IK for pose: [%.3f, %.3f, %.3f]", 
                 ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions (radians)
    auto ik_solutions = g_robot->ik(desired_flange);

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

    // Use RobotState to map the 6-DOF solution to the potentially larger joint group
    moveit::core::RobotState state(robot_model_);
    if (ik_seed_state.size() == getJointNames().size())
    {
        state.setJointGroupPositions(getGroupName(), ik_seed_state);
    }
    else
    {
        state.setToDefaultValues();
    }
    // Create a map of joint names to the solution values
    std::map<std::string, double> joint_values;
    for (size_t i = 0; i < 6; ++i)
    {
        joint_values["J" + std::to_string(i + 1)] = best[i];
    }
    state.setVariablePositions(joint_values);

    // Copy the full joint group state to the output solution
    state.copyJointGroupPositions(getGroupName(), solution);


    // Log the selected solution in degrees for readability
    std::vector<double> solution_deg;
    for (double rad : solution) solution_deg.push_back(rad * 180.0 / M_PI);
    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "Selected IK solution (degrees): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
                solution_deg[0], solution_deg[1], solution_deg[2], solution_deg[3], solution_deg[4], solution_deg[5]);

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

    // Adjust for tip offset
    Eigen::Isometry3d desired_flange = target_pose * tip_offset_.inverse();

    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Multi-IK input pose: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                ik_pose.position.x, ik_pose.position.y, ik_pose.position.z,
                ik_pose.orientation.w, ik_pose.orientation.x, ik_pose.orientation.y, ik_pose.orientation.z);
    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Pose in base: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                pose_in_base.position.x, pose_in_base.position.y, pose_in_base.position.z,
                pose_in_base.orientation.w, pose_in_base.orientation.x, pose_in_base.orientation.y, pose_in_base.orientation.z);

    // Log the latest marker pose for comparison (debugging)
    {
        std::lock_guard<std::mutex> lock(marker_pose_mutex_);
        RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"),
                    "Latest marker pose in frame '%s': pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                    "cylinder_stand",
                    latest_marker_pose_.position.x, latest_marker_pose_.position.y, latest_marker_pose_.position.z,
                    latest_marker_pose_.orientation.w, latest_marker_pose_.orientation.x, latest_marker_pose_.orientation.y, latest_marker_pose_.orientation.z);
    }

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Computing multi-solution IK for pose: [%.3f, %.3f, %.3f]",
                 ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(desired_flange);

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

    RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
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

    // Adjust for tip offset
    Eigen::Isometry3d desired_flange = target_pose * tip_offset_.inverse();

    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Search-IK input pose: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                ik_pose.position.x, ik_pose.position.y, ik_pose.position.z,
                ik_pose.orientation.w, ik_pose.orientation.x, ik_pose.orientation.y, ik_pose.orientation.z);
    RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"), "Pose in base: pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                pose_in_base.position.x, pose_in_base.position.y, pose_in_base.position.z,
                pose_in_base.orientation.w, pose_in_base.orientation.x, pose_in_base.orientation.y, pose_in_base.orientation.z);

    // Log the latest marker pose for comparison (debugging)
    {
        std::lock_guard<std::mutex> lock(marker_pose_mutex_);
        RCLCPP_INFO(rclcpp::get_logger("crx_kinematics"),
                    "Latest marker pose in frame '%s': pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                    "cylinder_stand",
                    latest_marker_pose_.position.x, latest_marker_pose_.position.y, latest_marker_pose_.position.z,
                    latest_marker_pose_.orientation.w, latest_marker_pose_.orientation.x, latest_marker_pose_.orientation.y, latest_marker_pose_.orientation.z);
    }

    RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                 "Searching for IK solutions (timeout: %.3f) for pose: [%.3f, %.3f, %.3f]",
                 timeout, ik_pose.position.x, ik_pose.position.y, ik_pose.position.z);

    // Get all IK solutions
    auto ik_solutions = g_robot->ik(desired_flange);

    if (ik_solutions.empty())
    {
        RCLCPP_WARN(kinematics::KinematicsBase::LOGGER,
                    "No IK solutions found");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
        return false;
    }

    // Prepare joint limits
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
        if (limits.size() != 6) return true;
        for (size_t i=0;i<6;++i)
        {
            if (q[i] < limits[i].first - 1e-6 || q[i] > limits[i].second + 1e-6)
                return false;
        }
        return true;
    };

    // Compute costs and filter valid solutions
    struct SolutionWithCost {
        std::array<double,6> sol;
        double cost;
    };
    std::vector<SolutionWithCost> valid_solutions;
    for (const auto& sol : ik_solutions)
    {
        if (!within_limits(sol))
            continue;
        // Compute FK to tip
        Eigen::Isometry3d T_tool = g_robot->fk(sol);
        Eigen::Isometry3d T = T_tool * tip_offset_;
        const Eigen::Vector3d dp = T.translation() - target_pose.translation();
        const Eigen::Matrix3d R_err = T.linear().transpose() * target_pose.linear();
        const double angle = std::acos(std::min(1.0, std::max(-1.0, (R_err.trace() - 1.0) / 2.0)));
        double cost = dp.squaredNorm() + 0.5 * angle * angle;
        // Seed bias if provided
        if (ik_seed_state.size() == 6)
        {
            double seed_cost = 0.0;
            for (size_t i = 0; i < 6; ++i)
            {
                const double d = sol[i] - ik_seed_state[i];
                seed_cost += d * d;
            }
            cost += 0.05 * seed_cost;
        }
        valid_solutions.push_back({sol, cost});
    }

    // Sort by cost
    std::sort(valid_solutions.begin(), valid_solutions.end(), [](const SolutionWithCost& a, const SolutionWithCost& b) {
        return a.cost < b.cost;
    });

    // Call the callback for each valid solution in order
    if (!valid_solutions.empty())
    {
        solution.clear();
        for (double angle : valid_solutions[0].sol)
            solution.push_back(angle);

        // For the best solution, log the achieved pose
        Eigen::Isometry3d T_tool = g_robot->fk(valid_solutions[0].sol);
        Eigen::Isometry3d T = T_tool * tip_offset_;
        const Eigen::Vector3d dp = T.translation() - target_pose.translation();
        const Eigen::Matrix3d R_err = T.linear().transpose() * target_pose.linear();
        const double angle = std::acos(std::min(1.0, std::max(-1.0, (R_err.trace() - 1.0) / 2.0)));
        const Eigen::Quaterniond q_target(target_pose.linear());
        const Eigen::Quaterniond q_best(T.linear());
        RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                    "Target pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f]",
                    target_pose.translation().x(), target_pose.translation().y(), target_pose.translation().z(),
                    q_target.w(), q_target.x(), q_target.y(), q_target.z());
        RCLCPP_INFO(kinematics::KinematicsBase::LOGGER,
                    "Best   pos=[%.4f, %.4f, %.4f], quat=[%.4f, %.4f, %.4f, %.4f], dpos=%.6f, ang_err=%.6f",
                    T.translation().x(), T.translation().y(), T.translation().z(),
                    q_best.w(), q_best.x(), q_best.y(), q_best.z(), dp.norm(), angle);

        // Log the solution in degrees
        std::vector<double> solution_deg;
        for (double rad : solution) solution_deg.push_back(rad * 180.0 / M_PI);
        RCLCPP_INFO(kinematics::KinematicsBase::LOGGER, "IK solution (degrees): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f], cost=%.6f",
                    solution_deg[0], solution_deg[1], solution_deg[2], solution_deg[3], solution_deg[4], solution_deg[5],
                    valid_solutions[0].cost);

        error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
        solution_callback(ik_pose, solution, error_code);
        
        RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                     "Called solution callback for best solution, cost=%.6f",
                     valid_solutions[0].cost);
    }

    return !valid_solutions.empty();
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
    // Use MoveIt's RobotState FK so we stay consistent with the
    // RobotModel and joint ordering that the rest of MoveIt uses.
    const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(getGroupName());
    if (!jmg)
    {
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                     "getPositionFK: could not get JointModelGroup '%s'",
                     getGroupName().c_str());
        return false;
    }

    if (joint_angles.size() != jmg->getVariableCount())
    {
        RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                     "getPositionFK: expected %u joint variables, got %lu",
                     jmg->getVariableCount(), joint_angles.size());
        return false;
    }

    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    state.setJointGroupPositions(jmg, joint_angles);

    poses.clear();
    poses.reserve(link_names.size());

    for (const auto& link_name : link_names)
    {
        const moveit::core::LinkModel* link = robot_model_->getLinkModel(link_name);
        if (!link)
        {
            RCLCPP_ERROR(kinematics::KinematicsBase::LOGGER,
                         "getPositionFK: unknown link '%s'", link_name.c_str());
            return false;
        }

        const Eigen::Isometry3d& T = state.getGlobalLinkTransform(link);
        geometry_msgs::msg::Pose pose = tf2::toMsg(T);

        RCLCPP_DEBUG(kinematics::KinematicsBase::LOGGER,
                     "FK(%s): pos=[%.3f, %.3f, %.3f], quat=[%.3f, %.3f, %.3f, %.3f]",
                     link_name.c_str(),
                     pose.position.x, pose.position.y, pose.position.z,
                     pose.orientation.w, pose.orientation.x,
                     pose.orientation.y, pose.orientation.z);

        poses.push_back(pose);
    }

    return true;
}

void CRXKinematicsPlugin::markerUpdateCallback(const visualization_msgs::msg::InteractiveMarkerUpdate::SharedPtr msg)
{
    if (!msg->poses.empty())
    {
        std::lock_guard<std::mutex> lock(marker_pose_mutex_);
        latest_marker_pose_ = msg->poses[0].pose;
    }
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
