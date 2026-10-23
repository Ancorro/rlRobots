#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)  # this folder, before any cd
# Second AWS box (A10G 24 GB, spot): Isaac Sim 5.1 + Isaac Lab 2.3.2 core + unitree_sim_isaaclab, then the G1 Dex3
# benchmark at 1, 64, 512, 2048 robots (same as box 1's isaac_setup.sh + isaac_lab2.sh + isaac_run.sh).
set -x
sudo chown ubuntu /opt/dlami/nvme
W=/opt/dlami/nvme/w; mkdir -p $W; cd $W
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH OMNI_KIT_ACCEPT_EULA=YES
sudo apt-get update -qq && sudo apt-get install -y -qq cmake build-essential git-lfs unzip >/dev/null
git clone -q https://github.com/unitreerobotics/unitree_sim_isaaclab.git
git clone -q https://github.com/isaac-sim/IsaacLab.git && (cd IsaacLab && git checkout -q v2.3.2)
git clone -q https://github.com/unitreerobotics/unitree_sdk2_python
git clone -q https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
(cd unitree_sim_isaaclab && git submodule update --init --depth 1 -q && bash fetch_assets.sh > $W/isaac-assets.log 2>&1)
(cd cyclonedds && mkdir -p build install && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install >/dev/null && cmake --build . --target install -j8 >/dev/null)
export CYCLONEDDS_HOME=$W/cyclonedds/install
uv venv -q --python 3.11 venv-isaac
P=$W/venv-isaac/bin/python
uv pip install -q --python $P torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -q --python $P "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install -q --python $P "setuptools<81" wheel pip
uv pip install -q --python $P --no-build-isolation flatdict==4.0.1
for s in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do uv pip install -q --python $P --no-build-isolation -e IsaacLab/source/$s || echo FAILED-$s; done
uv pip install -q --python $P -e unitree_sdk2_python
(cd unitree_sim_isaaclab && sed -i 's/type:.*/type: isaacsim/' teleimager/cam_config_server.yaml && uv pip install -q --python $P -r requirements.txt && uv pip install -q --python $P -e teleimager)
echo SETUP-DONE
