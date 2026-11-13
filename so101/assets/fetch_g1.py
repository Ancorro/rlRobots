"""Get the Unitree G1 humanoid model (with Dex3 hands) from MuJoCo Menagerie into so101/assets/unitree_g1/ (gitignored,
~38 MB of meshes).

Source: github.com/google-deepmind/mujoco_menagerie, folder unitree_g1, pinned commit below (the same commit the SO-101
model was vendored from). Licence: BSD-3-Clause (unitree_g1/LICENSE, copied along).

Usage (standard library + git):
    python so101/assets/fetch_g1.py                    # copy from a local Menagerie clone if there is one, else sparse-clone
    python so101/assets/fetch_g1.py --from=PATH        # copy from this unitree_g1 folder
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie"
COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
DEST = Path(__file__).parent / "unitree_g1"
LOCAL = Path.home() / "mujoco" / "menagerie" / "unitree_g1"  # where the owner's PC keeps its Menagerie clone


def copy(src):
    shutil.copytree(src, DEST, dirs_exist_ok=True)
    print(f"copied {src} -> {DEST}")


def main():
    src = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--from=")), None)
    if src is None and (LOCAL / "g1_with_hands.xml").exists():
        src = LOCAL
    if src is not None:
        copy(src)
        return
    with tempfile.TemporaryDirectory() as tmp:  # only the unitree_g1 folder, at the pinned commit
        git = ["git", "-C", tmp]
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, tmp], check=True)
        subprocess.run([*git, "sparse-checkout", "set", "unitree_g1"], check=True)
        subprocess.run([*git, "checkout", COMMIT], check=True)
        copy(Path(tmp) / "unitree_g1")


if __name__ == "__main__":
    main()
