// Python access to the analytic solver (CRXRobot) without MoveIt: a planner that asks thousands
// of reach questions (my_fanuc's push pre-filter) calls it in-process, where /compute_ik is a
// service round trip per pose and the pure-Python port (crx_kinematics_py) ~22 ms a solve.
// The plugin's frame handling (base height, flange extension, ROS/pendant tool rotation, the
// round-trip and joint-limit checks) is the caller's: this is the bare solver.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <stdexcept>
#include <string>

#include "crx_kinematics/robot.hpp"

namespace py = pybind11;
using crx_kinematics::CRXRobot;
using crx_kinematics::RobotNameEnum;

namespace
{
RobotNameEnum robot_name(const std::string& name)
{
    static const std::map<std::string, RobotNameEnum> names = {
        { "crx3ia", RobotNameEnum::crx3ia },       { "crx5ia", RobotNameEnum::crx5ia },
        { "crx10ia", RobotNameEnum::crx10ia },     { "crx10ia_l", RobotNameEnum::crx10ia_l },
        { "crx20ia_l", RobotNameEnum::crx20ia_l }, { "crx30ia", RobotNameEnum::crx30ia },
    };
    const auto it = names.find(name);
    if (it == names.end())
    {
        throw std::invalid_argument("unknown CRX model '" + name + "'");
    }
    return it->second;
}
}  // namespace

PYBIND11_MODULE(crx_kinematics_cpp, m)
{
    m.doc() = "The CRX analytic IK/FK (Abbes and Poisson 2024), as the MoveIt plugin uses it.";
    py::class_<CRXRobot>(m, "CRXRobot")
        .def(py::init([](const std::string& name, bool couple_j2_j3) {
                 return CRXRobot(robot_name(name), couple_j2_j3);
             }),
             py::arg("name"), py::arg("couple_j2_j3") = true)
        .def(
            "fk",
            [](const CRXRobot& r, const std::array<double, 6>& q) -> Eigen::Matrix4d {
                return r.fk(q).matrix();
            },
            py::arg("joint_values"), "Tool pose in R0 (4x4) for joint values (rad).")
        .def(
            "ik",
            [](const CRXRobot& r, const Eigen::Matrix4d& T) {
                Eigen::Isometry3d iso;
                iso.matrix() = T;
                return r.ik(iso);
            },
            py::arg("T_R0_tool"), "Every IK solution (rad) for a tool pose in R0 (4x4), unfiltered.");
}
