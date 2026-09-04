# Persistent Docker image for this repo's robotics/simulation experiments

Built after repeatedly re-installing the same packages from scratch in throwaway
`ros:humble` containers across several experiments (ROS2 live tests, Gazebo, the
RoboGuard-inspired LTL check) -- each fresh `apt-get`/`conda install` pass took several
real minutes. This image caches all of that so future experiments start in seconds.

## Build (once, or after changing `Dockerfile.robotics`)

```bash
docker build -f docker/Dockerfile.robotics -t dense-armor-robotics .
```

## Use

```bash
docker run --rm -v "<path to a scripts folder>:/ws:ro" dense-armor-robotics bash /ws/<script>.sh
```

`WORKDIR` is already `/ws`, ROS2 Humble is already sourced in every shell.

## What's inside

- **ROS2 Humble** (base image) -- `rclpy`, this repo's own `dense_armor_ros` package
  dependencies.
- **Gazebo (Ignition Fortress)** via `ros-humble-ros-gz-sim`/`-bridge`/`-image` --
  the real simulator, `ign gazebo` (NOT `gz sim` -- that rename is a later Gazebo
  Harmonic thing, confirmed directly against what this apt package actually installs).
- **`xacro`, `robot_state_publisher`** -- for spawning real robot models.
- **`huggingface_hub`, `pandas`, `pyarrow`** (system pip) -- for the real LeRobot
  dataset downloads used throughout this repo's robot-command experiments.
- **`dense-armor`** (system pip, real PyPI release) -- Dense-Armor's own promoted
  detectors/filters (`streaming`, `cusum`, `rate_limiter`, `cbf_filter`), so they can
  run against real live ROS2/Gazebo data directly inside this image.
- **Miniconda + real `spot`** at `/opt/miniconda`, reachable via the full path
  `/opt/miniconda/bin/python3` -- deliberately NOT on `PATH` by default, to avoid
  shadowing ROS2's own system Python (`rclpy` needs the system interpreter
  specifically). PyPI's own package literally named `spot` is an unrelated
  "DotCloud environment loader" namespace collision -- confirmed directly, never
  used; the real LTL/omega-automata library only exists on conda-forge.

## Real, disclosed limits

- Linux-only usage pattern (this machine has no NVIDIA GPU -- SAFER-Splat's own
  Gaussian-Splatting perception pipeline needs CUDA and isn't part of this image;
  see `docs/geometric_cbf_filter_real_joint_commands.md` for why that's a real,
  not incidental, gap).
- Not published to a registry -- lives only in this machine's local Docker image
  cache. Portable to another machine only by rebuilding from `Dockerfile.robotics`
  here, not by pulling a pre-built image.
