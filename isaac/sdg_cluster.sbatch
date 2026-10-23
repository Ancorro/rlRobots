#!/bin/bash
# Plan #2 first pass: Isaac Lab's G1 locomanipulation imitation pipeline (docs teleop_imitation.rst "Demo 2"), stock
# steering-wheel task, to check it runs on the cluster before we make demos for our objects.
#   STEP=gen sbatch isaac/sdg_cluster.sbatch   # download NVIDIA's annotated demos, generate 1,000 (isaaclab_mimic)
#   STEP=bc  sbatch isaac/sdg_cluster.sbatch   # robomimic BC on the generated set
#   STEP=nav sbatch isaac/sdg_cluster.sbatch   # walking between tables on top (locomanipulation_sdg); KIT= drops the
#                                              # mobility_gen kit extension (nav segfaulted at startup with it, 2026-10-18);
#                                              # KIT= LATE=1 enables it after the app starts (isaac/nav_late_ext.py: hangs)
#                                              # KIT= SHIM=1 loads only its path planner (isaac/nav_planner_shim.py)
#   STEP=carry [WALK=vx,vy,wz,s;...] [ROLL=10] [TAG=] [RESET=1] sbatch isaac/sdg_cluster.sbatch   # BC pick, our scripted walk, BC
#                                              # place (isaac/bc_carry.py); out isaac/sdg/carry-<TAG>.json
#   STEP=play [CKPT=model_epoch_2000.pth] [ROLL=50] sbatch isaac/sdg_cluster.sbatch   # BC rollouts, docs' settings
#   STEP=navbc [NAVIN='nav300_*'] [EPOCHS=1000] sbatch isaac/sdg_cluster.sbatch   # nav data -> robot-frame BC set (isaac/nav_to_bc.py),
#                                              # then robomimic BC (isaac/nav_bc_rnn.json); out isaac/sdg/navbc/
#   STEP=navmod [PLACE=bc|replay] [CARRY=frozen|demo] [HIP=0.8] [SETTLE=0] [RAMP=0] [ROLL=20] [TAG=] sbatch isaac/sdg_cluster.sbatch   # manipulation BC picks, generator's
#                                              # path follower walks, BC or recorded release places (isaac/nav_modular_eval.py)
#   STEP=placebc STOCK=full|tail [TAG=] sbatch isaac/sdg_cluster.sbatch   # place BC: stock demos (whole, or from the +6 cm lift)
#                                              # plus the nav drop-offs in the start table's frame (isaac/nav_to_place.py), stock
#                                              # BC config; out isaac/sdg/robomimic-place-<TAG>/; navmod PLACERUN=<that run dir> uses it
#   STEP=boxgen|boxbc|boxplay BOX_SIZE=x,y,z BOX_MASS=kg BOX_POS=x,y,z [BOXT=box1] [TRIALS=1000] sbatch isaac/sdg_cluster.sbatch
#                                              # the wheel task with a box (isaac/box_cfgs.py, box_tasks.py): Mimic demos from
#                                              # the wheel's annotated demos, robomimic BC, rollouts; out sdg/generated_<BOXT>.hdf5
#   STEP=boxreplay BOX_SIZE=.. sbatch isaac/sdg_cluster.sbatch   # wheel demos replayed with the wheel and the box (isaac/box_replay.py)
#   STEP=boxbcplay BOXT=box3 [CKPT=] [ROLL=20] [HANDOFF=steps] [NEAR=m] [TAG=] ...   # BC rollouts with how far each gets; HANDOFF: script sets down (isaac/box_bcplay.py)
#   STEP=boxbc BOXT=box3 MERGE='scripted_box3-*.hdf5' ...   # merge recorded demo files, then BC on them
#   STEP=boxscript BOX_GOAL=x,y [SCRIPTARGS='--roll 0,30'] [RECORD=1] sbatch ...   # scripted two-hand box demos (isaac/box_scripted.py)
#   STEP=boxrl BOX_GOAL=x,y [NE=4096] [ITERS=300] [RLARGS=--walk] TAG= sbatch ...   # PPO on the box task, AGILE legs, batch IK (isaac/box_rl.py)
#   STEP=boxwalk BOX_SIZE=.. BOX_POS=.. [ROLL=5] [SEED=] [TAG=] sbatch ...   # scripted box pick, walk to the other table, set down (isaac/box_walk.py)
#   BOX_SIZED=1: boxscript records object_size, boxbc trains the size-aware BC (BoxSized task), boxwalk passes --sized
#     PICK=bc BOXT=box4 [CKPT=] [HANDOFF=100] [RECORD=1]: the box BC grabs and lifts instead of the script
#   STEP=probe sbatch isaac/sdg_cluster.sbatch   # wheel size, mass and where the hands hold it (isaac/box_probe.py)
#   STEP=navplay [NCKPT=path.pth] [ROLL=20] [TAG=] sbatch isaac/sdg_cluster.sbatch   # nav BC rollouts in the nav scene, TRACE=1 per-step npz,
#                                              # judged (isaac/nav_bc_eval.py); out isaac/sdg/navplay-<TAG>.json
#SBATCH --job-name=g1-sdg
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --constraint=a40|l40s|rtx8000|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/hpc/share/%u/isaac/sdg-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
I=$SHARE/isaac; D=$I/sdg; P=$I/pylib-mimic; STEP=${STEP:-gen}
CAMS=${CAMS-"--enable_cameras"}; NAV=${NAV:-generated_dataset_g1_locomanipulation_sdg}  # nav: CAMS= turns cameras off
KIT=${KIT-"--kit_args=--enable isaacsim.replicator.mobility_gen"}
WALK=${WALK:-"-0.2,0,0,2;0,0,0,1;0.2,0,0,2;0,0,0,1"}; TAG=${TAG:-$SLURM_JOB_ID}
RUN=${RUN:-20261018185839}; CKPT=${CKPT:-model_epoch_2000.pth}; ROLL=${ROLL:-50}  # play: the bc run, its checkpoint, rollouts
mkdir -p $D $P
B=$D/robomimic/Isaac-PickPlace-Locomanipulation-G1-Abs-v0/bc_rnn_low_dim_g1  # robomimic asks y/n on stdin if it exists
[ $STEP = bc ] && [ -d $B ] && mv $B $B.old-$(date +%Y%m%d%H%M%S)
N=$D/navbc/nav_bc_rnn; NAVIN=${NAVIN:-"nav300_*"}
[ $STEP = navbc ] && [ -d $N ] && mv $N $N.old-$(date +%Y%m%d%H%M%S)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
URL=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/pick_place_datasets/dataset_annotated_g1_locomanip.hdf5
[ -s $D/dataset_annotated_g1_locomanip.hdf5 ] || curl -fsSL -o $D/dataset_annotated_g1_locomanip.hdf5 $URL || { echo "download failed"; exit 2; }
ls -l $D
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_GIT_PYTHON_REFRESH=quiet APPTAINERENV_PYTHONUNBUFFERED=1
# box task (isaac/box_cfgs.py): size "x,y,z" m, mass kg, centre "x,y,z"; BOXT = data and run name
for v in BOX_SIZE BOX_MASS BOX_POS BOX_WALL BOX_YAW BOX_GOAL BOX_SIZED BOX_SHAPE IK_WAIST IK_STEP; do [ -n "${!v:-}" ] && export APPTAINERENV_$v="${!v}"; done
BOXT=${BOXT:-box1}; BB=$D/robomimic-$BOXT/Isaac-PickPlace-Locomanipulation-G1-Box${BOX_SIZED:+Sized}-Abs-v0/bc_rnn_low_dim_g1
# robomimic and isaaclab_mimic in their own folder (no deps: the container's torch stays as it is)
export APPTAINERENV_PYTHONPATH=$P:$I/IsaacLab/source/isaaclab_mimic:$I/pylib
apptainer exec --cleanenv --writable --nv --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
cd $I/IsaacLab
/isaac-sim/python.sh -c 'import robomimic' 2>/dev/null || /isaac-sim/python.sh -m pip install -q --no-deps --target $P \
  https://github.com/ARISE-Initiative/robomimic/archive/refs/tags/v0.4.0.tar.gz h5py termcolor tensorboardX 2>&1 | tail -3
/isaac-sim/python.sh -c 'import IPython' 2>/dev/null || /isaac-sim/python.sh -m pip install -q --target $P ipython 2>&1 | tail -3
# isaaclab_mimic's own requirements (setup.py) and robomimic's pure-Python ones, no deps
/isaac-sim/python.sh -c 'import ipywidgets, robomimic.algo' 2>/dev/null || /isaac-sim/python.sh -m pip install -q --no-deps --target $P \
  ipywidgets==8.1.5 comm widgetsnbextension jupyterlab_widgets tomli psutil tqdm imageio imageio-ffmpeg diffusers 2>&1 | tail -3
/isaac-sim/python.sh -c 'import robomimic.algo, IPython, ipywidgets; import robomimic; print(\"imports ok\", robomimic.__version__)' || exit 3
L=$I/sdg-run-$SLURM_JOB_ID.log  # expanded outside: the container has no SLURM_JOB_ID
case $STEP in
gen) /isaac-sim/python.sh scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --headless --num_envs 20 \
       --generation_num_trials 1000 --enable_pinocchio --input_file $D/dataset_annotated_g1_locomanip.hdf5 \
       --output_file $D/generated_dataset_g1_locomanip.hdf5 > \$L 2>&1 ;;
bc)  /isaac-sim/python.sh scripts/imitation_learning/robomimic/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --algo bc \
       --normalize_training_actions --dataset $D/generated_dataset_g1_locomanip.hdf5 --log_dir $D/robomimic > \$L 2>&1 ;;
nav) /isaac-sim/python.sh ${LATE:+$I/nav_late_ext.py} ${SHIM:+$I/nav_planner_shim.py} scripts/imitation_learning/locomanipulation_sdg/generate_data.py --device cpu --headless \
       ${KIT:+\"$KIT\"} --task=Isaac-G1-SteeringWheel-Locomanipulation \
       --dataset $D/generated_dataset_g1_locomanip.hdf5 --num_runs ${RUNS:-1} --lift_step 60 --navigate_step 130 --enable_pinocchio \
       --output_file $D/$NAV.hdf5 $CAMS > \$L 2>&1 ;;
play) NF=\$(cat $B/$RUN/logs/normalization_params.txt); MIN=\$(echo \"\$NF\" | awk '/min/{print \$2}'); MAX=\$(echo \"\$NF\" | awk '/max/{print \$2}')
     /isaac-sim/python.sh scripts/imitation_learning/robomimic/play.py --device cpu --headless --enable_pinocchio        --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --num_rollouts $ROLL --horizon 400 --norm_factor_min \$MIN        --norm_factor_max \$MAX --checkpoint $B/$RUN/models/$CKPT > \$L 2>&1 ;;
carry) NF=\$(cat $B/$RUN/logs/normalization_params.txt); MIN=\$(echo \"\$NF\" | awk '/min/{print \$2}'); MAX=\$(echo \"\$NF\" | awk '/max/{print \$2}')
     /isaac-sim/python.sh $I/bc_carry.py --device cpu --headless --enable_pinocchio --num_rollouts $ROLL --walk=\"$WALK\" ${RESET:+--reset_rnn} \
       --norm_factor_min \$MIN --norm_factor_max \$MAX --checkpoint $B/$RUN/models/$CKPT --out $D/carry-$TAG.json > \$L 2>&1 ;;
placebc) /isaac-sim/python.sh $I/nav_to_place.py $D/place-$TAG.hdf5 --stock_file $D/generated_dataset_g1_locomanip.hdf5 \
       --stock ${STOCK:-tail} $D/$NAVIN.hdf5 > \$L 2>&1 && \
     /isaac-sim/python.sh scripts/imitation_learning/robomimic/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --algo bc \
       --normalize_training_actions --dataset $D/place-$TAG.hdf5 --log_dir $D/robomimic-place-$TAG >> \$L 2>&1 ;;
boxgen) /isaac-sim/python.sh $I/box_tasks.py scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cpu --headless \
       --num_envs 20 --generation_num_trials ${TRIALS:-1000} --enable_pinocchio --task Isaac-Locomanipulation-G1-Box-Abs-Mimic-v0 \
       --input_file $D/dataset_annotated_g1_locomanip.hdf5 --output_file $D/generated_$BOXT.hdf5 > \$L 2>&1 ;;
boxbc) ${MERGE:+/isaac-sim/python.sh scripts/tools/merge_hdf5_datasets.py --input_files \$(ls $D/$MERGE) --output_file $D/generated_$BOXT.hdf5 > \$L 2>&1 &&}
       /isaac-sim/python.sh $I/box_tasks.py scripts/imitation_learning/robomimic/train.py --task Isaac-PickPlace-Locomanipulation-G1-Box${BOX_SIZED:+Sized}-Abs-v0 \
       --algo bc --normalize_training_actions --dataset $D/generated_$BOXT.hdf5 --log_dir $D/robomimic-$BOXT >> \$L 2>&1 ;;
boxplay) R=\$(ls -d $BB/2026* | tail -1); NF=\$(cat \$R/logs/normalization_params.txt); MIN=\$(echo \"\$NF\" | awk '/min/{print \$2}'); MAX=\$(echo \"\$NF\" | awk '/max/{print \$2}')
     /isaac-sim/python.sh $I/box_tasks.py scripts/imitation_learning/robomimic/play.py --device cpu --headless --enable_pinocchio \
       --task Isaac-PickPlace-Locomanipulation-G1-Box-Abs-v0 --num_rollouts ${ROLL:-20} --horizon ${HORIZON:-400} --norm_factor_min \$MIN \
       --norm_factor_max \$MAX --checkpoint \$R/models/model_epoch_2000.pth > \$L 2>&1 ;;
boxbcplay) R=\$(ls -d $BB/2026* | tail -1); NF=\$(cat \$R/logs/normalization_params.txt); MIN=\$(echo \"\$NF\" | awk '/min/{print \$2}'); MAX=\$(echo \"\$NF\" | awk '/max/{print \$2}')
     /isaac-sim/python.sh $I/box_tasks.py $I/box_bcplay.py --device ${DEV:-cpu} --ik ${IK:-pink} --headless --enable_pinocchio \
       --task Isaac-PickPlace-Locomanipulation-G1-Box${BOX_SIZED:+Sized}-Abs-v0 --num_rollouts ${ROLL:-20} --horizon ${HORIZON:-800} --norm_factor_min \$MIN \
       --norm_factor_max \$MAX --checkpoint \$R/models/${CKPT:-model_epoch_2000.pth} ${HANDOFF:+--handoff $HANDOFF --near ${NEAR:-0.04}} --dataset $D/generated_dataset_g1_locomanip.hdf5 ${CLEAN:+--clean_reset $CLEAN} ${JIT:+--jitter $JIT} ${RECORD:+--record} --out $D/boxbcplay-$TAG.json > \$L 2>&1 ;;
boxreplay) for o in wheel box; do /isaac-sim/python.sh $I/box_replay.py --object \$o --dataset $D/generated_dataset_g1_locomanip.hdf5 \
       --demos ${ROLL:-5} --device cpu --headless --enable_pinocchio; done > \$L 2>&1 ;;
boxscript) /isaac-sim/python.sh $I/box_scripted.py --dataset $D/generated_dataset_g1_locomanip.hdf5 --trials ${ROLL:-4} --envs ${ENVS:-1} --ik ${IK:-pink} \
       ${SCRIPTARGS:-} ${RECORD:+--record $D/scripted_$BOXT.hdf5} --out $D/boxscript-$TAG.json --device ${DEV:-cpu} --headless --enable_pinocchio > \$L 2>&1 ;;
boxwalk) PB=; if [ -n \"${PICK:-}\" ]; then R=\$(ls -d $BB/2026* | tail -1); NF=\$(cat \$R/logs/normalization_params.txt)
       PB=\"--pick bc --pick_ckpt \$R/models/${CKPT:-model_epoch_2000.pth} --pick_min \$(echo \"\$NF\" | awk '/min/{print \$2}') --pick_max \$(echo \"\$NF\" | awk '/max/{print \$2}') ${HANDOFF:+--handoff $HANDOFF} ${BOX_SIZED:+--sized}\"; fi
     /isaac-sim/python.sh $I/box_walk.py --trials ${ROLL:-5} --seed ${SEED:-0} \$PB ${RECORD:+--record} ${SCRIPTARGS:-} --out $D/boxwalk-$TAG.json --device cpu --headless \
       --task=Isaac-G1-SteeringWheel-Locomanipulation --dataset $D/generated_dataset_g1_locomanip.hdf5 --enable_pinocchio > \$L 2>&1 ;;
boxrl) /isaac-sim/python.sh $I/box_rl.py train --num_envs ${NE:-4096} --iters ${ITERS:-300} --out $D/boxrl-$TAG \
       --dataset $D/generated_dataset_g1_locomanip.hdf5 ${RLARGS:-} > \$L 2>&1 ;;
probe) /isaac-sim/python.sh $I/box_probe.py --dataset $D/generated_dataset_g1_locomanip.hdf5 --device cpu --headless --enable_pinocchio > \$L 2>&1 ;;
navbc) /isaac-sim/python.sh $I/nav_to_bc.py $D/navbc.hdf5 $D/$NAVIN.hdf5 > \$L 2>&1 && \
     /isaac-sim/python.sh -c \"import json; c=json.load(open('$I/nav_bc_rnn.json')); e=${EPOCHS:-1000}; c['train']['num_epochs']=e
c['experiment']['save']['every_n_epochs']=min(100, e); c['train']['output_dir']='$D/navbc'; json.dump(c, open('$D/nav_bc_rnn-$TAG.json', 'w'))\" && \
     /isaac-sim/python.sh $P/robomimic/scripts/train.py --config $D/nav_bc_rnn-$TAG.json --dataset $D/navbc.hdf5 >> \$L 2>&1 ;;
navplay) C=${NCKPT:-\$(ls -v $N/*/models/*.pth | tail -1)}; echo checkpoint \$C
     /isaac-sim/python.sh $I/nav_bc_eval.py --checkpoint \$C --norm $D/navbc.norm.json --trials ${ROLL:-20} \
       --out $D/navplay-$TAG.json ${TRACE:+--trace $D/navplay-$TAG.npz} --device cpu --headless --task=Isaac-G1-SteeringWheel-Locomanipulation \
       --dataset $D/generated_dataset_g1_locomanip.hdf5 --enable_pinocchio > \$L 2>&1 ;;
navmod) NF=\$(cat $B/$RUN/logs/normalization_params.txt); MIN=\$(echo \"\$NF\" | awk '/min/{print \$2}'); MAX=\$(echo \"\$NF\" | awk '/max/{print \$2}')
     /isaac-sim/python.sh $I/nav_modular_eval.py --pick_ckpt $B/$RUN/models/$CKPT --pick_min \$MIN --pick_max \$MAX \
       ${PLACERUN:+--place_ckpt \$(ls -v $PLACERUN/models/*.pth | tail -1) --place_min \$(awk '/min/{print \$2}' $PLACERUN/logs/normalization_params.txt) --place_max \$(awk '/max/{print \$2}' $PLACERUN/logs/normalization_params.txt)} \
       --place ${PLACE:-bc} --carry ${CARRY:-frozen} --hip ${HIP:-0.8} --reach ${REACH:-0.2} --settle ${SETTLE:-0} --ramp ${RAMP:-0} --trials ${ROLL:-20} --out $D/navmod-$TAG.json --device cpu --headless \
       --task=Isaac-G1-SteeringWheel-Locomanipulation --dataset $D/generated_dataset_g1_locomanip.hdf5 --enable_pinocchio > \$L 2>&1 ;;
esac
rc=\$?
case $STEP in gen) out=$D/generated_dataset_g1_locomanip.hdf5;; boxgen) out=$D/generated_$BOXT.hdf5;; boxwalk) out=$D/boxwalk-$TAG.json;; boxbcplay) out=$D/boxbcplay-$TAG.json;; nav) out=$D/$NAV.hdf5;; carry) out=$D/carry-$TAG.json;; navplay) out=$D/navplay-$TAG.json;; navmod) out=$D/navmod-$TAG.json;; *) out=;; esac
[ -n \"\$out\" ] && [ ! -s \"\$out\" ] && { echo \"no output \$out\"; rc=6; }
[ $STEP = navbc ] && ! ls $N/*/models/*.pth >/dev/null 2>&1 && { echo \"no nav bc checkpoint\"; rc=7; }
[ $STEP = placebc ] && ! ls $D/robomimic-place-$TAG/*/*/*/models/*.pth >/dev/null 2>&1 && { echo \"no place bc checkpoint\"; rc=7; }
[ $STEP = boxbc ] && ! ls $BB/*/models/*.pth >/dev/null 2>&1 && { echo \"no box bc checkpoint\"; rc=7; }
[ $STEP = bc ] && ! ls $B/*/models/*.pth >/dev/null 2>&1 && { echo \"no bc checkpoint\"; rc=7; }
grep -aiE 'box_walk|box_bcplay|BOX_PROBE|BOX_REPLAY|BOX_SCRIPT|BOX_PHASE|nav_modular|nav_bc_eval|kept|wrote|bc_carry|nav_planner_shim|success|generated|trials|epoch|norm|error|Traceback' \$L | grep -v WARN | tail -15; echo SDG-DONE step=$STEP rc=\$rc; exit \$rc"
