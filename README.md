# IsaacSim Tactile Environment for FlexiTac

This repository contains tools for running FlexiTac in IsaacSim, including:

- **trajectory replay with tactile visualization**
- **a browser-based Viser interface for interactive control**

## Installation

### 1. Install Isaac Sim

First, install Isaac Sim by following the official documentation:

[Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html)

### 2. Clone the repository

This repository uses **Git LFS** for large asset files. Please install Git LFS before cloning, otherwise some mesh and USD assets may be downloaded as pointer files instead of real content.

```bash
sudo apt install git-lfs
git lfs install
git clone xxxxxxxxxx
```

Follow the [IsaacLab instruction guideline](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) setup instructions in this repo.



```bash
pip uninstall -y opencv-python-headless
pip install opencv-python==4.11.0.86
pip install viser 
```

## Replay a Recorded Trajectory

Launch the tactile replay viewer with a saved dataset episode:

```bash
cd ~/repos/IsaacLab

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
cd ~/repos/IsaacLab

./isaaclab.sh -p Isaacsim_tactile_env/viser_interface.py \
    --headless \
    --enable_cameras
```

Then open the following in your browser:

```text
http://localhost:8080
```