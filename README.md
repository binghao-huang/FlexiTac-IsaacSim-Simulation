# IsaacSim Tactile Environment for FlexiTac

This repository contains tools for running FlexiTac in IsaacSim, including:

- **trajectory replay with tactile visualization**
- **a browser-based Viser interface for interactive control**

## Installation

### 1. Install Isaac Sim
Skip this step if Isaac Sim is already installed.

Follow the official NVIDIA documentation:  
[Isaac Sim Installation Guide](https://docs.isaacsim.nvidia.com/5.1.0/installation/quick-install.html)

1. Download Isaac Sim 5.1.0 or later:  
   [Isaac Sim 5.1.0 Linux Download](https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip)
2. Extract the downloaded archive into an `isaac-sim` directory.
3. On Linux, run:

```bash
./post_install.sh
./isaac-sim.selector.sh
```

### 3. Install IsaacLab and dependencies

Set up IsaacLab by following the official installation guide:  
[IsaacLab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

Create and activate a conda environment:

```bash
conda create -n tactile_isaaclab python=3.11
conda activate tactile_isaaclab
```

Install additional dependencies:

```bash
pip uninstall -y opencv-python-headless
pip install opencv-python==4.11.0.86
pip install viser
```

## Replay a Recorded Trajectory

Launch the tactile replay viewer with a saved dataset episode:

```bash
cd ~/FlexiTac-IsaacSim-Simulation

./isaaclab.sh -p Isaacsim_tactile_env/reply_with_tactile.py \
    --dataset_npz Isaacsim_tactile_env/data/dataset_train.npz \
    --normalization_pth Isaacsim_tactile_env/data/dataset_normalizer.npz \
    --episode_idx 0 \
    --replay_key joint_states \
    --steps_per_frame 3
```

## Launch the Interactive Web Interface

Start the Viser-based teleoperation interface:

```bash
cd ~/FlexiTac-IsaacSim-Simulation

./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
    --headless \
    --enable_cameras
```

Then open the following in your browser:

```text
http://localhost:8080
```