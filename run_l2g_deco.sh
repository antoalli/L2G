#!/bin/bash
CODE_ROOT=$(pwd)  # change this if script is not anymore in main folder!
DATA_ROOT=${CODE_ROOT}/data  # path to datasets directory
SN_DATA=${DATA_ROOT}/ShapeNetSem-8
YCB8_DATA=${DATA_ROOT}/YCB-8
YCB76_DATA=${DATA_ROOT}/YCB-76

# params
NN=100  # nn at sampled contact for feat. aggr.
M=500  # num grasps / sampled contacts
ENCODER="deco"
TRAIN_SPLIT="train"
SEED=21996
CHECKPOINTS_DIR=./experiments
LR=0.0001
WD=0.0001
LR_STEP=100
DECO_CONFIG=deco/deco_config.yaml
TRAIN_EPOCHS=500

# training on sn-8
cd "$CODE_ROOT" || exit
python train_l2g.py \
  --seed "$SEED" \
  --epochs $TRAIN_EPOCHS \
  --checkpoints_dir $CHECKPOINTS_DIR \
  --data_root $SN_DATA \
  --feat $ENCODER \
  --deco_config $DECO_CONFIG \
  --lr_step $LR_STEP \
  --neigh_size $NN \
  --sampled_grasps $M \
  --use_angle_feat True \
  --lr $LR \
  --wd $WD \
  --split $TRAIN_SPLIT;

# test all scenarios [sn->sn, sn->ycb-8, sn->ycb-76]
TEST_EPOCHS=(500)
EXP_DIR=${CHECKPOINTS_DIR}/deco_neigh-size${NN}_use-angle-feat_True_lambda0.1_samp-grasp${M}_match-policy-soft_posi-ratio-0.3/opt-adam_lr${LR}_lr-step${LR_STEP}_wd${WD}_epochs${TRAIN_EPOCHS}_seed${SEED}_${TRAIN_SPLIT}
SN_TEST="${EXP_DIR}/checkpoints/test"
YCB8_TEST="${EXP_DIR}/checkpoints/ycb8_test"
YCB76_TEST="${EXP_DIR}/checkpoints/ycb76_test"
mkdir -p $SN_TEST
mkdir -p $YCB8_TEST
mkdir -p $YCB76_TEST
cd "$CODE_ROOT" || EXIT
for EPOCH in "${TEST_EPOCHS[@]}"; do
  curr_ckt="${EXP_DIR}/checkpoints/epoch_${EPOCH}.pth"
  echo "Testing ckt ${curr_ckt} on ShapeNetSem-8"
  python test_l2g.py --test_type test  --resume "$curr_ckt" --data_root $SN_DATA;
  echo "Testing ckt ${curr_ckt} on YCB-8"
  python test_l2g.py --test_type ycb8_test --resume "$curr_ckt" --data_root $YCB8_DATA;
  echo "Testing ckt ${curr_ckt} on YCB-76"
  python test_l2g.py --test_type ycb76_test  --resume "$curr_ckt" --views "0,1,2,3,4,5,6,7,8,9" --data_root $YCB76_DATA;
done

# evaluate all scenarios [sn->sn, sn->ycb-8, sn->ycb-76]
# will produce rule-based and simulation results
cd "${CODE_ROOT}/grasp-evaluator" || EXIT
# ShapeNetSem-8
python -m gpnet_eval --dataset_root "$SN_DATA" --object_models_dir "${SN_DATA}/urdf" --test_dir $SN_TEST;
# YCB-8
python -m gpnet_eval --dataset_root "$YCB8_DATA" --object_models_dir "${YCB8_DATA}/urdf" --test_dir $YCB8_TEST;
# YCB-76
python -m gpnet_eval --dataset_root "$YCB76_DATA" --object_models_dir "${YCB76_DATA}/urdf" --test_dir $YCB76_TEST --no_cov;
# finish
cd "$CODE_ROOT" || EXIT
