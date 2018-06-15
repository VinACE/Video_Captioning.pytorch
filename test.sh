#!/usr/bin/env bash

func_cst_testing()
{
    	CUDA_VISIBLE_DEVICES=$$GPU_ID python test.py    --model_file $(word 1,$^) \
                                                        --test_label_h5 output/metadata/msrvtt_test_sequencelabel.h5 \
                                                        --test_cocofmt_file output/metadata/msrvtt_test_cocofmt.json \
                                                        --test_feat_h5 $(patsubst %,output/feature/msrvtt_test_%_mp1.h5,resnet c3d mfcc category)\
                                                        --beam_size 5 \
                                                        --language_eval 1 \
                                                        --test_seq_per_img 20 \
                                                        --test_batch_size 64 \
                                                        --loglevel INFO \
                                                        --result_file $@
}

func_cst_testing