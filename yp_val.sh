TRAIN_MODEL_NAME=$1
TEST_MODEL_NAME=$2

if [[ -z "${TEST_MODEL_NAME}" ]]; then
    TEST_MODEL_NAME=${TRAIN_MODEL_NAME}
    OUT_MODEL_NAME=${TRAIN_MODEL_NAME}
else
    OUT_MODEL_NAME=${TRAIN_MODEL_NAME}-${TEST_MODEL_NAME}
fi
echo "Train set: ${TRAIN_MODEL_NAME} test set: ${TEST_MODEL_NAME} output: ${OUT_MODEL_NAME}"
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Using ${NUM_GPUS} GPU(s) for training."

python -m tools.data_preprocess \
    --data_root dataset/${TRAIN_MODEL_NAME} \
    --interval 1 --sam_model sam --data_type val

echo "Training GS model for ${TRAIN_MODEL_NAME}"
python -m tools.train_gs_models \
    --data dataset/${TRAIN_MODEL_NAME} \
    --output output/gs_models/${TRAIN_MODEL_NAME} \
    --gpus ${NUM_GPUS} --threads 4

    
python -m tools.gen_real_matches \
   --data dataset/${TEST_MODEL_NAME} \
   --ckpt_root output/gs_models/${TRAIN_MODEL_NAME} \
   --save output/gs_models/${TRAIN_MODEL_NAME} --interval 1 \
   --data_type val

python -m tools.merge --config config/preprocess/merge_annotation_val_match_yp.yaml train_model_name=${TRAIN_MODEL_NAME} test_model_name=${TEST_MODEL_NAME} out_model_name=${OUT_MODEL_NAME} 
python -m tools.merge --config config/preprocess/merge_annotation_val_align_yp.yaml train_model_name=${TRAIN_MODEL_NAME} test_model_name=${TEST_MODEL_NAME} out_model_name=${OUT_MODEL_NAME} 

python test.py --testing_type joint --config config/experiment/test_joint_yp.yaml train_model_name=${TRAIN_MODEL_NAME} test_model_name=${TEST_MODEL_NAME} out_model_name=${OUT_MODEL_NAME} 