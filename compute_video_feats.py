'''
Load saliency feature map, extracted vgg16
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import time

import cv2
import h5py
import numpy as np
import skimage
import torch
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms as trn
from misc.encoder.AppearanceEncoder import *
from misc.encoder.MotionEncoder import *
from misc.utils import *

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def parse_opt():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--type', type=str, default='renset', help="[renset, motion, audio, category, c3d]")
    parser.add_argument('--frame_size', type=int, default=224)   # Frame size
    # Frame
    parser.add_argument('--feat_size', type=int, default=4096) # Frame feature
    parser.add_argument('--frame_sample_rate', type=int, default=10)  # Video sample rate
    parser.add_argument('--max_frames', type=int, default=20)  # Number of frames
    parser.add_argument('--num_frames', type=int, default=20)  # Number of frames
    # Models
    parser.add_argument('--resnet_checkpoint', type=str, default='./misc/encoder/pytorch-resnet/resnet101.pth')
    parser.add_argument('--c3d_checkpoint', type=str, default='./datasets/models/c3d.pickle')
    # MSR_VTT
    parser.add_argument('--msrvtt_video_root', type=str, default='./datasets/MSR-VTT/TrainValVideo/')
    parser.add_argument('--msrvtt_anno_json_path', type=str, default='./datasets/MSR-VTT/train_val_videodatainfo.json')
    # MSVD
    parser.add_argument('--msvd_video_root', type=str, default='./datasets/MSVD/youtube_videos_id')
    parser.add_argument('--msvd_csv_path', type=str, default='./datasets/MSVD/MSR_Video_Description_Corpus.csv') # MSR_Video_Description_Corpus_refine
    parser.add_argument('--msvd_video_name2id_map', type=str, default='./datasets/MSVD/youtube_mapping.txt')
    parser.add_argument('--msvd_anno_json_path', type=str, default='./datasets/MSVD/annotations.json')
    # Output
    parser.add_argument('--feat_h5', type=str, default='output/metadata/msrvtt_resnet')
    args = parser.parse_args()

    msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
    args.msrvtt_train_range = (0, 6513 - 1)
    args.msrvtt_val_range = (6512, 6512 + 2990 - 1)
    args.msrvtt_test_range = (6512 + 2990, 6512 + 2990 + 497 - 1)
    # msvd_video_sort_lambda = lambda x: int(x[3:-4])
    msvd_video_sort_lambda = lambda x: int(x[5:-4])
    args.msvd_train_range = (0, 1200 - 1)
    args.msvd_val_range = (1200, 1200 + 100 - 1)
    args.msvd_test_range = (1300, 1300 + 470 - 1)

    args.video_root = args.msrvtt_video_root if args.dataset=='msrvtt' else args.msvd_video_root
    args.video_sort_lambda = msrvtt_video_sort_lambda if args.dataset == 'msrvtt' else msvd_video_sort_lambda
    args.anno_json_path = args.msrvtt_anno_json_path if args.dataset == 'msrvtt' else args.msvd_anno_json_path
    args.train_range = args.msrvtt_train_range if args.dataset == 'msrvtt' else args.msvd_train_range
    args.val_range = args.msrvtt_val_range if args.dataset == 'msrvtt' else args.msvd_val_range
    args.test_range = args.msrvtt_test_range if args.dataset == 'msrvtt' else args.msvd_test_range
    return args

def sample_frames(opt, video_path, train=True):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    indices = np.linspace(8, frame_count - 7, opt.max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # Copy one channel gray image three times to from RGB image
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def preprocess_frame_full(I, aencoder, resize):
    # handle grayscale input images
    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)

    I = I.astype('float32') / 255.0
    I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
    I = Variable(preprocess(I), volatile=True)
    fc = aencoder(I, resize)
    return fc.data.cpu().float().numpy()


def extract_resnet_features(opt, encoder, resize=False):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'test', 'val'], [opt.train_range[1], opt.test_range[1], opt.val_range[1]]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '_' + keys[i] + '_' + opt.type + '.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (nvideos, opt.num_frames, opt.feat_size), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (nvideos,), dtype='int')
        with tqdm(total=values[i] + 1) as pbar:
            for i in xrange(values[i] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(i) + '.mp4')

                frame_list, clip_list, frame_count = sample_frames(opt, video_path, train=True)
                feats = np.zeros((opt.max_frames, opt.feat_size), dtype='float32')
                if resize:
                    frame_list = np.array([preprocess_frame(x) for x in frame_list])
                    frame_list = frame_list.transpose((0, 3, 1, 2))
                    frame_list = Variable(torch.from_numpy(frame_list), volatile=True).cuda()
                    af = encoder(frame_list, resize)
                    feats[:frame_count, :] = af.data.cpu().numpy()
                else:
                    af = []
                    for x in frame_list:
                        af.append(preprocess_frame_full(x, encoder, resize))
                    af = np.array(af)
                    feats[:frame_count, :] = af

                dataset_feats[i] = feats
                dataset_lens[i] = frame_count

def extract_motion_features(opt, encoder, resize=False):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'test', 'val'], [opt.train_range[1], opt.test_range[1], opt.val_range[1]]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '_' + keys[i] + '_' + opt.type + '.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (nvideos, opt.num_frames, opt.feat_size), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (nvideos,), dtype='int')
        with tqdm(total=values[i] + 1) as pbar:
            for i in xrange(values[i] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(i) + '.mp4')

                frame_list, clip_list, frame_count = sample_frames(opt, video_path, train=True)
                clip_list = np.array([[resize_frame(x, 112, 112) for x in clip] for clip in clip_list])
                clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
                clip_list = Variable(torch.from_numpy(clip_list), volatile=True).cuda()
                dataset_feats[i] = encoder(clip_list)
                dataset_lens[i] = frame_count

if __name__ == '__main__':
    opt = parse_opt()
    #build_msvd_annotation(opt)
    if opt.type == 'renset':
        extract_resnet_features(opt, AppearanceEncoder(opt).eval().cuda())
    elif opt.type == 'motion':
        extract_motion_features(opt, MotionEncoder(opt).eval().cuda())
    else:
        print("You need select one type!")

