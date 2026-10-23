# Humanoid pick, carry and place: research notes (2026-10-22)

Gathered for the goal "walk around environments, move objects of varied shape and size". Each item was read at the link given; HumanPlus, OmniH2O and BEHAVIOR/RoboCasa were not checked.

## Closest to our setup

- **VIRAL** (He et al.), [arXiv 2511.15200](https://arxiv.org/abs/2511.15200): G1 + Dex3 in Isaac Lab, walk / place / grasp / lift / turn. A privileged PPO (proximal policy optimization) teacher sends commands to a frozen HOMIE whole-body controller.
  - Delta actions on velocity, yaw rate, arm joints and fingers.
  - Fingers on low gains (Kp 0.5, Kd 0.1).
  - Release reward: penalty on finger-object contact force within 0.3 m of the tray.
  - Reference state initialization from 200 demos: under 10% success without it, about 95% with it.
  - Use for us: reset box PPO from scripted demo states, including holding the box at the target; delta actions on top of the IK (inverse kinematics) targets.
- **Multi-humanoid pickup** (Pandit et al.), [arXiv 2609.17824](https://arxiv.org/html/2609.17824): gripperless Digit robots pinch a box with both hands, Isaac Lab, frozen locomotion.
  - Hand points at plus or minus half-width relative to the torso; palm normals flush with the faces.
  - Assisted-contact curriculum: the box is held up until two contacts are detected.
  - Release by an idle/track flag; pickup above 97%.
- **Sim-to-real dexterous humanoid** (Lin et al.), [arXiv 2502.20396](https://arxiv.org/html/2502.20396): two-hand box lift. Reward pulls the fingertips to "contact sticker" markers on the box; where the markers sit sets the grasp. Giving the policy the object position mattered (10/10 vs 0/10).
- **ResMimic**, [arXiv 2510.05070](https://arxiv.org/html/2510.05070), [code](https://github.com/amazon-far/ResMimic): G1 box carry. A virtual PD (proportional-derivative) force on the object that fades during training, plus contact rewards on the arms and torso so the robot hugs the box.
- **Humanoid Hanoi**, [arXiv 2602.13850](https://arxiv.org/html/2602.13850): table removed 45 steps after contact, so the policy has to lift. Place demos are reversed pickups.
- **CoorDex**, [arXiv 2606.23680](https://arxiv.org/html/2606.23680): "grasped" means mean fingertip distance under 0.10 m plus at least 2 contacts over 1 N, one of them the thumb.
- **Whole-Body Dexterity**, [arXiv 2610.22184](https://arxiv.org/html/2610.22184v1): G1 + Dex3 cradles objects (a basketball) against the arms and torso. Real robot: carry 9/10, place 7/10.

## Physics

- **No rolling friction.** PhysX has [Coulomb sliding friction only](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html), and the [NVIDIA forum](https://forums.developer.nvidia.com/t/friction-not-working-on-rolling-sphere/202483) has no fix. Workaround: angular damping on the object (we use 20).
- **Twisting friction.** `torsional_patch_radius` adds twisting friction (we use 0.01).
- **Grip friction.** Combine mode "max" lets the grip friction win.
- **Dexsuite.** Ramps gravity up through an automatic curriculum, instead of using a lift reward.

## Code

- **Locomanipulation SDG** (synthetic data generation), [docs](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/humanoids_imitation.html): one environment only.
- **SkillGen**, [docs](https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/skillgen.html): Franka only; needs cuRobo and 24 GB of GPU memory.
- **AGILE** [paper](https://arxiv.org/html/2603.20147) and [repo](https://github.com/nvidia-isaac/WBC-AGILE).
- **ManiSkill [StackCube](https://raw.githubusercontent.com/haosulab/ManiSkill/main/mani_skill/envs/tasks/tabletop/stack_cube.py):** success means placed and still and released, and the reward pays for an open gripper and a still object once the object is on target. A pattern for our learned set-down.

## Applied here

- **Grip lower on the box (walk16).** Our squeeze sat 5 cm below the top of a 20 cm box, above its centre of mass, and boxes tipped. walk16 tries `--grip_below` 0.10 and 0.13.
- **Angular damping 20 on rolling shapes (walk_grasp).**
