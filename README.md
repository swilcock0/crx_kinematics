# crx_kinematics

<div align="center">

[![Humble](https://github.com/danielcranston/crx_kinematics/actions/workflows/humble.yml/badge.svg?branch=master)](https://github.com/danielcranston/crx_kinematics/actions/workflows/humble.yml)
[![Jazzy](https://github.com/danielcranston/crx_kinematics/actions/workflows/jazzy.yml/badge.svg?branch=master)](https://github.com/danielcranston/crx_kinematics/actions/workflows/jazzy.yml)
[![Kilted](https://github.com/danielcranston/crx_kinematics/actions/workflows/kilted.yml/badge.svg?branch=master)](https://github.com/danielcranston/crx_kinematics/actions/workflows/kilted.yml)
[![Rolling](https://github.com/danielcranston/crx_kinematics/actions/workflows/rolling.yml/badge.svg?branch=master)](https://github.com/danielcranston/crx_kinematics/actions/workflows/rolling.yml)

</div>

This repo hosts C++ and Python code implementing FK/IK for the Fanuc CRX series. The implementation closely follows _[Geometric Approach for Inverse Kinematics of the FANUC CRX Collaborative Robot](https://www.mdpi.com/2218-6581/13/6/91)_ by Abbes and Poisson (2024).

<img src="readme_images/ik.gif">

Compared to general optimization-based IK solvers like KDL, the implementation in this repo

* Deterministically finds all IK solutions
* Has near-zero dependencies (numpy+scipy for Python, Eigen for C++)
* Is fast (C++ implementation runs in ~50 μs on a Intel Core i7-13650HX)

The approach reduces the IK problem to a 1-D search for zeros over a scalar function. See [DERIVATION.md](DERIVATION.md) for an overview of the approach.

The C++ package also hosts a Moveit 2 Kinematics plugin that is compatible out of the box with [the official Fanuc URDF descriptions](https://github.com/FANUC-CORPORATION/fanuc_description/).

# Examples

The Python package comes with a interactive demo:

```bash
ros2 launch crx_kinematics_py demo.launch.py run_rviz:=true
```

In terms of API:

```cpp
#include "crx_kinematics/robot.hpp"

int main(int argc, char** argv)
{
    auto robot = crx_kinematics::CRXRobot();  // Defaults to CRX-10iA
    Eigen::Isometry3d pose = robot.fk({ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });

    std::vector<std::array<double, 6>> joint_solutions = robot.ik(pose);
}
```

```python
from crx_kinematics_py.robot import CRXRobot
import numpy as np

robot = CRXRobot()  # Defaults to CRX-10iA
pose: np.ndarray = robot.fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

joint_solutions, debug_data = robot.ik(pose)
```

## Moveit 2 Kinematics Plugin

To use the Moveit 2 Kinematics plugin, build the `crx_kinematics` package in your workspace, then edit the [kinematics.yaml](https://github.com/FANUC-CORPORATION/fanuc_driver/blob/v2.0.0/fanuc_moveit_config/config/kinematics.yaml#L7) file of your Moveit 2 config package to use the plugin:

```diff
manipulator:
-  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
+  kinematics_solver: crx_kinematics/CRXKinematicsPlugin
```

The plugin also works with custom URDFs, provided their base and tip frames, as well as the name of the URDF, match the equivalent URDF from Fanuc.

Note that, while the code in this repo supports all active ROS distros, the official Fanuc driver only supports Humble and Jazzy. If you want to use the _controllers_ provided in the Fanuc driver repo (e.g. `ScaledJointTrajectoryController`) together with this plugin, you are restricted to using a distro supported by the Fanuc driver repo.

### Plugin parameters

All parameters are read from `kinematics.yaml` under the `<group_name>.` namespace (e.g. `manipulator.solution_selection`).

#### IK solution selection

Because the solver always finds **all** valid IK solutions (up to 16), you can choose how the best one is selected:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solution_selection` | string | `"distance"` | Criterion used to rank solutions. See options below. |
| `min_manipulability` | double | `0.0` | Reject solutions whose manipulability score falls below this floor (applies to `manip1` and `manip2` only). Disabled when `≤ 0`. |
| `seed_bias` | double | `0.0` | Weight multiplied by the L1 seed distance and added to the manipulability / clearance score. Allows trading off maximising the metric against staying close to the seed. `0` disables. |

The `solution_selection` values mirror the scoring modes of [TRAC-IK](https://traclabs.com/projects/trac-ik/):

| Value | Criterion | Notes |
|-------|-----------|-------|
| `"distance"` | Minimise L1 joint-space distance to the seed state | **Default.** Cheapest to compute; best trajectory continuity. |
| `"manip1"` | Maximise √(det(**J J**ᵀ)) — Yoshikawa manipulability index | Analogous to TRAC-IK's `Manip1`. Larger values mean further from a singularity. |
| `"manip2"` | Maximise σ_min / σ_max — inverse condition number of **J**, in [0, 1] | Analogous to TRAC-IK's `Manip2`. Scale-invariant; `1` is perfectly isotropic. |
| `"clearance"` | Maximise minimum distance to self-collision | Requires `collision_detector: fcl` (see below). |

Example — pick the most manipulable solution, but discard any that are nearly singular and keep a slight bias toward the seed:

```yaml
manipulator:
  kinematics_solver: crx_kinematics/CRXKinematicsPlugin
  solution_selection: manip1
  min_manipulability: 0.05
  seed_bias: 0.1
```

#### Self-collision checking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `self_collision_check` | bool | `false` | Reject any IK solution that puts the robot in self-collision. Costs one collision query per candidate. |
| `collision_detector` | string | `"fcl"` | Collision library to use. `"fcl"` or `"bullet"`. `"clearance"` solution selection requires `"fcl"` (Bullet's `distanceSelf` is an unimplemented stub in MoveIt). |

#### Tool / flange offset

The plugin can shift the IK target from the URDF tip frame to a physical tool control point (TCP), so a single URDF can serve multiple tools without modification:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flange_extension` | double | `0.0` | Translation along the tip frame's +Z axis, in metres. Convenient for a tool mounted axially on the flange. |
| `tool_offset_xyz` | list\<double\> (3) | `[0, 0, 0]` | Full XYZ translation of the TCP relative to the tip frame, in metres. Applied **on top of** `flange_extension`. |
| `tool_offset_rpy` | list\<double\> (3) | `[0, 0, 0]` | RPY rotation of the TCP frame relative to the tip frame, in radians (extrinsic X-Y-Z). |

Example — 100 mm tool extension with a 90° rotation about Z:

```yaml
manipulator:
  kinematics_solver: crx_kinematics/CRXKinematicsPlugin
  flange_extension: 0.1
  tool_offset_rpy: [0.0, 0.0, 1.5708]
```

# Cloning and building

Using standard ROS 2 steps:

```bash
# Clone
cd ~/your_workspace/src
git clone git@github.com:danielcranston/crx_kinematics.git

# Install binary dependencies via rosdep
rosdep install --from-paths crx_kinematics --ignore-src -y

# Build
cd ~/your_workspace
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to crx_kinematics crx_kinematics_py

# Test
colcon test --event-handlers console_cohesion+ --packages-select crx_kinematics crx_kinematics_py
```

# License

MIT
