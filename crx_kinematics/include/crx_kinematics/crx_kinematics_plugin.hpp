#pragma once

#include <moveit/kinematics_base/kinematics_base.h>
#include <Eigen/Geometry>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <visualization_msgs/msg/interactive_marker_update.hpp>
#include <mutex>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace crx_kinematics
{

class CRXKinematicsPlugin : public kinematics::KinematicsBase
{
  public:
    virtual bool initialize(rclcpp::Node::SharedPtr const& node,
                            moveit::core::RobotModel const& robot_model,
                            std::string const& group_name,
                            std::string const& base_frame,
                            std::vector<std::string> const& tip_frames,
                            double search_discretization) override final;

    virtual bool getPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                               const std::vector<double>& ik_seed_state,
                               std::vector<double>& solution,
                               moveit_msgs::msg::MoveItErrorCodes& error_code,
                               const kinematics::KinematicsQueryOptions& options =
                                   kinematics::KinematicsQueryOptions()) const override final;

    virtual bool getPositionIK(const std::vector<geometry_msgs::msg::Pose>& ik_poses,
                               const std::vector<double>& ik_seed_state,
                               std::vector<std::vector<double>>& solutions,
                               kinematics::KinematicsResult& result,
                               const kinematics::KinematicsQueryOptions& options =
                                   kinematics::KinematicsQueryOptions()) const override final;

    virtual bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                  const std::vector<double>& ik_seed_state,
                                  double timeout,
                                  std::vector<double>& solution,
                                  moveit_msgs::msg::MoveItErrorCodes& error_code,
                                  const kinematics::KinematicsQueryOptions& options =
                                      kinematics::KinematicsQueryOptions()) const override final;
    virtual bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                  const std::vector<double>& ik_seed_state,
                                  double timeout,
                                  const std::vector<double>& consistency_limits,
                                  std::vector<double>& solution,
                                  moveit_msgs::msg::MoveItErrorCodes& error_code,
                                  const kinematics::KinematicsQueryOptions& options =
                                      kinematics::KinematicsQueryOptions()) const override final;
    virtual bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                  const std::vector<double>& ik_seed_state,
                                  double timeout,
                                  std::vector<double>& solution,
                                  const IKCallbackFn& solution_callback,
                                  moveit_msgs::msg::MoveItErrorCodes& error_code,
                                  const kinematics::KinematicsQueryOptions& options =
                                      kinematics::KinematicsQueryOptions()) const override final;
    virtual bool searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                  const std::vector<double>& ik_seed_state,
                                  double timeout,
                                  const std::vector<double>& consistency_limits,
                                  std::vector<double>& solution,
                                  const IKCallbackFn& solution_callback,
                                  moveit_msgs::msg::MoveItErrorCodes& error_code,
                                  const kinematics::KinematicsQueryOptions& options =
                                      kinematics::KinematicsQueryOptions()) const override final;

    virtual bool getPositionFK(const std::vector<std::string>& link_names,
                               const std::vector<double>& joint_angles,
                               std::vector<geometry_msgs::msg::Pose>& poses) const override final;

    virtual const std::vector<std::string>& getJointNames() const override final;
    virtual const std::vector<std::string>& getLinkNames() const override final;

  private:
    std::vector<std::string> joint_names_;
    std::vector<std::string> link_names_;
    std::string planning_frame_;
    std::string robot_base_frame_;

    Eigen::Isometry3d tip_offset_ = Eigen::Isometry3d::Identity();
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // For debugging marker discrepancies
    rclcpp::Subscription<visualization_msgs::msg::InteractiveMarkerUpdate>::SharedPtr marker_subscriber_;
    geometry_msgs::msg::Pose latest_marker_pose_;
    mutable std::mutex marker_pose_mutex_;
    void markerUpdateCallback(const visualization_msgs::msg::InteractiveMarkerUpdate::SharedPtr msg);
};

}  // namespace crx_kinematics