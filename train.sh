#! /bin/bash

func_training_xe()
{
    DATASET="msrvtt2016"
    CKPT_NAME=XE
    CKPT_DIR=output/model/${CKPT_NAME}
	mkdir -p ${CKPT_DIR}
	CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --train_label_h5 output/metadata/${DATASET}_train_sequencelabel.h5 \
                                                    --val_label_h5 output/metadata/${DATASET}_val_sequencelabel.h5 \
                                                    --test_label_h5 output/metadata/${DATASET}_test_sequencelabel.h5 \
                                                    --train_cocofmt_file output/metadata/${DATASET}_train_cocofmt.json \
                                                    --val_cocofmt_file output/metadata/${DATASET}_val_cocofmt.json \
                                                    --test_cocofmt_file output/metadata/${DATASET}_test_cocofmt.json \
                                                    --train_bcmrscores_pkl output/metadata/${DATASET}_train_evalscores.pkl \
                                                    --train_feat_h5 output/feature/msrvtt_train_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_train_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_train_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_train_category_mp1.h5 \
                                                    --val_feat_h5   output/feature/msrvtt_val_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_val_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_val_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_val_category_mp1.h5 \
                                                    --test_feat_h5  output/feature/msrvtt_test_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_test_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_test_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_test_category_mp1.h5 \
                                                    --beam_size 5 --max_patience 50 --eval_metric CIDEr --print_log_interval 20 --language_eval 1 \
                                                    --max_epochs 50 --rnn_size 512 --train_seq_per_img 20 --test_seq_per_img 20 --batch_size 64 \
                                                    --test_batch_size 64 --learning_rate 0.0001 --lr_update 200 --save_checkpoint_from 1 --num_chunks 1 \
                                                    --train_cached_tokens output/metadata/${DATASET}_train_ciderdf.pkl \
                                                    --ss_k 100 --use_rl_after 0 --ss_max_prob 0.25 --use_rl 0 --use_mixer 0 --mixer_from -1 --use_cst 0 \
                                                    --scb_captions 0 --scb_baseline 1 --loglevel INFO --model_type concat --use_eos 0 \
                                                    --model_file ${CKPT_DIR}/model.pth \
                                                    --start_from No \
                                                    --result_file ${CKPT_DIR}/model_test.json \
                                                    2>&1 | tee ${CKPT_DIR}/run.log
}

func_training_wxe()
{
    DATASET="msrvtt2016"
    CKPT_NAME=WXE
    CKPT_DIR=output/model/${CKPT_NAME}
	mkdir -p ${CKPT_DIR}
	CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --train_label_h5 output/metadata/${DATASET}_train_sequencelabel.h5 \
                                                    --val_label_h5 output/metadata/${DATASET}_val_sequencelabel.h5 \
                                                    --test_label_h5 output/metadata/${DATASET}_test_sequencelabel.h5 \
                                                    --train_cocofmt_file output/metadata/${DATASET}_train_cocofmt.json \
                                                    --val_cocofmt_file output/metadata/${DATASET}_val_cocofmt.json \
                                                    --test_cocofmt_file output/metadata/${DATASET}_test_cocofmt.json \
                                                    --train_bcmrscores_pkl output/metadata/${DATASET}_train_evalscores.pkl \
                                                    --train_feat_h5 output/feature/msrvtt_train_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_train_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_train_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_train_category_mp1.h5 \
                                                    --val_feat_h5   output/feature/msrvtt_val_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_val_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_val_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_val_category_mp1.h5 \
                                                    --test_feat_h5  output/feature/msrvtt_test_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_test_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_test_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_test_category_mp1.h5 \
                                                    --beam_size 5 --max_patience 50 --eval_metric CIDEr --print_log_interval 20 --language_eval 1 \
                                                    --max_epochs 50 --rnn_size 512 --train_seq_per_img 20 --test_seq_per_img 20 --batch_size 64 \
                                                    --test_batch_size 64 --learning_rate 0.0001 --lr_update 200 --save_checkpoint_from 1 --num_chunks 1 \
                                                    --train_cached_tokens output/metadata/${DATASET}_train_ciderdf.pkl \
                                                    --ss_k 100 --use_rl_after 0 --ss_max_prob 0.25 --use_rl 1 --use_mixer 0 --mixer_from -1 --use_cst 1 \
                                                    --scb_captions 0 --scb_baseline 1 --loglevel INFO --model_type concat --use_eos 0 \
                                                    --model_file ${CKPT_DIR}/model.pth \
                                                    --start_from No \
                                                    --result_file ${CKPT_DIR}/model_test.json \
                                                    2>&1 | tee ${CKPT_DIR}/run.log
}

func_training_cst_ms_greedy()
{
    DATASET="msrvtt2016"
    CKPT_NAME=CST_MS_Greedy
    CKPT_DIR=output/model/${CKPT_NAME}
	mkdir -p ${CKPT_DIR}
	CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --train_label_h5 output/metadata/${DATASET}_train_sequencelabel.h5 \
                                                    --val_label_h5 output/metadata/${DATASET}_val_sequencelabel.h5 \
                                                    --test_label_h5 output/metadata/${DATASET}_test_sequencelabel.h5 \
                                                    --train_cocofmt_file output/metadata/${DATASET}_train_cocofmt.json \
                                                    --val_cocofmt_file output/metadata/${DATASET}_val_cocofmt.json \
                                                    --test_cocofmt_file output/metadata/${DATASET}_test_cocofmt.json \
                                                    --train_bcmrscores_pkl output/metadata/${DATASET}_train_evalscores.pkl \
                                                    --train_feat_h5 output/feature/msrvtt_train_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_train_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_train_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_train_category_mp1.h5 \
                                                    --val_feat_h5   output/feature/msrvtt_val_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_val_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_val_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_val_category_mp1.h5 \
                                                    --test_feat_h5  output/feature/msrvtt_test_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_test_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_test_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_test_category_mp1.h5 \
                                                    --beam_size 5 --max_patience 50 --eval_metric CIDEr --print_log_interval 20 --language_eval 1 \
                                                    --max_epochs 200 --rnn_size 512 --train_seq_per_img 20 --test_seq_per_img 20 --batch_size 64 \
                                                    --test_batch_size 64 --learning_rate 0.0001 --lr_update 200 --save_checkpoint_from 1 --num_chunks 1 \
                                                    --train_cached_tokens output/metadata/${DATASET}_train_ciderdf.pkl \
                                                    --ss_k 100 --use_rl_after 0 --ss_max_prob 0.25 --use_rl 1 --use_mixer 1 --mixer_from -1 --use_cst 0 \
                                                    --scb_captions 0 --scb_baseline 1 --loglevel INFO --model_type concat --use_eos 1 \
                                                    --model_file ${CKPT_DIR}/model.pth \
                                                    --start_from output/model/WXE \
                                                    --result_file ${CKPT_DIR}/model_test.json \
                                                    2>&1 | tee ${CKPT_DIR}/run.log
}

func_training_cst_ms_scb_v1()
{
    DATASET="msrvtt2016"
    CKPT_NAME=CST_MS_SCB_V1
    CKPT_DIR=output/model/${CKPT_NAME}
	mkdir -p ${CKPT_DIR}
	CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --train_label_h5 output/metadata/${DATASET}_train_sequencelabel.h5 \
                                                    --val_label_h5 output/metadata/${DATASET}_val_sequencelabel.h5 \
                                                    --test_label_h5 output/metadata/${DATASET}_test_sequencelabel.h5 \
                                                    --train_cocofmt_file output/metadata/${DATASET}_train_cocofmt.json \
                                                    --val_cocofmt_file output/metadata/${DATASET}_val_cocofmt.json \
                                                    --test_cocofmt_file output/metadata/${DATASET}_test_cocofmt.json \
                                                    --train_bcmrscores_pkl output/metadata/${DATASET}_train_evalscores.pkl \
                                                    --train_feat_h5 output/feature/msrvtt_train_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_train_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_train_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_train_category_mp1.h5 \
                                                    --val_feat_h5   output/feature/msrvtt_val_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_val_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_val_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_val_category_mp1.h5 \
                                                    --test_feat_h5  output/feature/msrvtt_test_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_test_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_test_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_test_category_mp1.h5 \
                                                    --beam_size 5 --max_patience 50 --eval_metric CIDEr --print_log_interval 20 --language_eval 1 \
                                                    --max_epochs 200 --rnn_size 512 --train_seq_per_img 20 --test_seq_per_img 20 --batch_size 64 \
                                                    --test_batch_size 64 --learning_rate 0.0001 --lr_update 200 --save_checkpoint_from 1 --num_chunks 1 \
                                                    --train_cached_tokens output/metadata/${DATASET}_train_ciderdf.pkl \
                                                    --ss_k 100 --use_rl_after 0 --ss_max_prob 0.25 --use_rl 1 --use_mixer 1 --mixer_from 1 --use_cst 1 \
                                                    --scb_captions 20 --scb_baseline 1 --loglevel INFO --model_type concat --use_eos 1 \
                                                    --model_file ${CKPT_DIR}/model.pth \
                                                    --start_from output/model/WXE \
                                                    --result_file ${CKPT_DIR}/model_test.json \
                                                    2>&1 | tee ${CKPT_DIR}/run.log
}


func_training_cst_ms_scb_v2()
{
    DATASET="msrvtt2016"
    CKPT_NAME=eCST_MS_SCB_V2
    CKPT_DIR=output/model/${CKPT_NAME}
	mkdir -p ${CKPT_DIR}
	CUDA_VISIBLE_DEVICES=$GPU_ID python train.py    --train_label_h5 output/metadata/${DATASET}_train_sequencelabel.h5 \
                                                    --val_label_h5 output/metadata/${DATASET}_val_sequencelabel.h5 \
                                                    --test_label_h5 output/metadata/${DATASET}_test_sequencelabel.h5 \
                                                    --train_cocofmt_file output/metadata/${DATASET}_train_cocofmt.json \
                                                    --val_cocofmt_file output/metadata/${DATASET}_val_cocofmt.json \
                                                    --test_cocofmt_file output/metadata/${DATASET}_test_cocofmt.json \
                                                    --train_bcmrscores_pkl output/metadata/${DATASET}_train_evalscores.pkl \
                                                    --train_feat_h5 output/feature/msrvtt_train_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_train_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_train_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_train_category_mp1.h5 \
                                                    --val_feat_h5   output/feature/msrvtt_val_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_val_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_val_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_val_category_mp1.h5 \
                                                    --test_feat_h5  output/feature/msrvtt_test_resnet_mp1.h5 \
                                                                    output/feature/msrvtt_test_c3d_mp1.h5 \
                                                                    output/feature/msrvtt_test_mfcc_mp1.h5.vggish \
                                                                    output/feature/msrvtt_test_category_mp1.h5 \
                                                    --beam_size 5 --max_patience 50 --eval_metric CIDEr --print_log_interval 20 --language_eval 1 \
                                                    --max_epochs 200 --rnn_size 512 --train_seq_per_img 20 --test_seq_per_img 20 --batch_size 64 \
                                                    --test_batch_size 64 --learning_rate 0.0001 --lr_update 200 --save_checkpoint_from 1 --num_chunks 1 \
                                                    --train_cached_tokens output/metadata/${DATASET}_train_ciderdf.pkl \
                                                    --ss_k 100 --use_rl_after 0 --ss_max_prob 0.25 --use_rl 1 --use_mixer 1 --mixer_from 1 --use_cst 1 \
                                                    --scb_captions 20 --scb_baseline 2 --loglevel INFO --model_type concat --use_eos 1 \
                                                    --model_file ${CKPT_DIR}/model.pth \
                                                    --start_from output/model/WXE \
                                                    --result_file ${CKPT_DIR}/model_test.json \
                                                    2>&1 | tee ${CKPT_DIR}/run.log
}

GPU_ID=$2 # Get gpu id
case "$1" in
    0) echo "Train XE model" && func_training_xe;;
    1) echo "Train CST_GT_None/WXE model" && func_training_wxe;;
    2) echo "Train CST_MS_Greedy model (using greedy baseline)" && func_training_cst_ms_greedy;;
    3) echo "Train CST_MS_SCB model (using SCB baseline, where SCB is computed from GT captions)" && func_training_cst_ms_scb_v1;;
    4) echo "Train CST_MS_SCB(*) model (using SCB baseline, where SCB is computed from model sampled captions)" && func_training_cst_ms_scb_v2;;
    *) echo "No input" ;;
esac