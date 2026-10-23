# Grasp training research (notes, 2026-10-08)

Grasp-training research, compiled from web searches (abstracts, project pages); numbers are the papers' own and not re-checked against PDFs. Different papers use different hands, objects and success tests, so their percentages are not comparable with each other or with ours.

## Where our grasp stands

Scripted top-down pinch + 8 learned settings chosen once per grasp (open loop), optional re-grasp after a failed 3 cm test lift. Object state from the simulator plus camera-like error; arm by damped least squares IK; trained on 6 primitive shapes. Test (realistic settings): v8 282/360 lifted, 302 with 2 retries; household objects 58/342 (77 with retries).

## What the field does

| Work | Method | Scale | Result (their test) |
|---|---|---|---|
| UniDexGrasp++ (ICCV 2023) | closed-loop RL policy for arm + hand; state-based teacher, then vision student; geometry-aware curriculum, generalist-specialist training | 3,000+ objects, table top | 85.4% train / 78.2% unseen objects (sim) |
| DextrAH-G (CoRL 2024), DextrAH-RGB (2024) | RL teacher with object state, distilled (DAgger) into a depth / stereo-RGB student; **geometric fabrics** controller keeps the arm off limits, itself and the table | 23-motor arm + hand, many objects | continuous fast grasp-and-transport on hardware |
| Isaac Lab Dexsuite (`Isaac-Dexsuite-Kuka-Allegro-Lift-v0`, in our Isaac Lab 2.3) | PPO on joint actions, contact-rich proprioception; builds on DextrAH and DexPBT | 512 envs, ~4 h one GPU | reliably lifts procedurally generated cuboids |
| DexGraspNet (2022) / 2.0 (CoRL 2024) | grasp datasets from simulation; 2.0 trains a diffusion grasp generator | 1.32M grasps on 5,355 objects; 427M grasps, 1,319 objects, cluttered | 2.0: ~91% real cluttered grasping, zero-shot |
| Get a Grip (CoRL 2024) | learned grasp evaluator scores candidate multi-finger grasps from vision | 3.5M grasps, 4.3K objects | beats analytic and generative baselines; quality falls with less data |
| Tactile slip control (e.g. SlipSense 2026) | detect slip from touch, re-grip | Tesollo hand | object loss prevented 95/100 |

Success tests used: DexGraspNet holds the grasp under gravity in all 6 axis directions for 100 steps (penetration <= 0.1 cm); others push with 0.5 x object weight along +-x, y, z; DexGraspNet 2.0 lifts with friction 0.2 on hands and objects (ours 0.5-1.0).

## What it means for us

1. **Closed-loop policy, not one-shot settings.** The working recipe is an RL policy that moves the arm and fingers every step, trained with full object state (teacher), then distilled into a camera-based student. Our one-shot settings plateau near 78-84%.
2. **The arm controller is a known problem with a known fix.** DextrAH's geometric fabrics handle joint limits and collisions (self, table) under the learned policy. Our biggest failure class, "arm off target", is that problem; our clamp/null-space attempt failed.
3. **Train on many objects.** Everyone trains on thousands (DexGraspNet, UniDexGrasp, Visual Dexterity set). Six primitive shapes explain much of the household gap (58/342).
4. **A ready starting point:** Isaac Lab's Dexsuite lift task runs on our Isaac Lab version. Porting it to the fixed-base G1 + Dex3 gives the standard PPO setup.
5. **Sensors.** Today object state is privileged (simulator truth plus noise); joints are what the G1 measures. The research path keeps the privileged teacher but trains a student on the G1's own sensors (head depth camera; Dex3 touch sensors later).

## Possible next steps

- A: Port Dexsuite lift to G1 + Dex3 (fixed base), PPO teacher with object state on primitives + a training split of the scanned objects; keep the realism settings; test on today's benchmarks.
- B: A collision- and limit-aware arm controller (fabrics-like) under the current skill, to remove "arm off target".
- C: Then distill A's teacher into a depth-camera student; tactile slip re-grip after that.

Sources: [UniDexGrasp++](https://arxiv.org/abs/2304.00464), [DextrAH-G](https://arxiv.org/abs/2407.02274), [DextrAH-RGB](https://arxiv.org/abs/2412.01791), [Isaac Lab environments 2.3](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments), [Dexsuite lift workflow](https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/dexsuite_lift/index.html), [Isaac Lab paper](https://arxiv.org/abs/2511.04831), [DexGraspNet](https://arxiv.org/abs/2210.02697), [DexGraspNet 2.0](https://arxiv.org/abs/2410.23004), [Get a Grip](https://arxiv.org/abs/2410.23701), [GraspGen](https://arxiv.org/abs/2507.13097) (parallel jaw), [Closing the Reality Gap](https://arxiv.org/abs/2601.02778), [AnyDexGrasp](https://arxiv.org/abs/2502.16420), [SlipSense](https://arxiv.org/abs/2609.15910), [CoorDex (G1 + hand)](https://arxiv.org/abs/2606.23680).
