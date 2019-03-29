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
import pdb
import cv2
import h5py
import numpy as np
import skimage
import torch
from tqdm import tqdm
from skimage.transform import resize
from torch.autograd import Variable
from torchvision import transforms as trn
from misc.encoder.AppearanceEncoder import *
from misc.encoder.MotionEncoder import *
from misc.utils import *
from misc.encoder.C3D import *
from misc.saliency import *
from misc.p3d.p3d_model import *


preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def parse_opt():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--type', type=str, default='saliency', help="[renset, motion, audio, category, c3d, p3d]")
    parser.add_argument('--frame_size', type=int, default=224)   # Frame size
    # Frame
    parser.add_argument('--feat_size', type=int, default=4096) # Frame feature
    parser.add_argument('--frame_sample_rate', type=int, default=10)  # Video sample rate
    parser.add_argument('--max_frames', type=int, default=20)  # Number of frames
    parser.add_argument('--num_frames', type=int, default=20)  # Number of frames
    # Models
    parser.add_argument('--resnet_checkpoint', type=str, default='/content/Video_Captioning.pytorch/misc/encoder/pytorch-resnet/resnet101-5d3b4d8f.pth')
    parser.add_argument('--c3d_checkpoint', type=str, default='./datasets/models/c3d.pickle')
    # MSR_VTT
    parser.add_argument('--msrvtt_video_root', type=str, default='/content/Video_Captioning.pytorch/datasets/msrvtt/videos/')
    parser.add_argument('--msrvtt_anno_json_path', type=str, default='./datasets/msrvtt/train_val_videodatainfo.json')
    # MSVD
    parser.add_argument('--msvd_video_root', type=str, default='./datasets/msvd/youtube_videos_id')
    parser.add_argument('--msvd_csv_path', type=str, default='./datasets/msvd/MSR_Video_Description_Corpus.csv') # MSR_Video_Description_Corpus_refine
    parser.add_argument('--msvd_video_name2id_map', type=str, default='./datasets/msvd/youtube_mapping.txt')
    parser.add_argument('--msvd_anno_json_path', type=str, default='./datasets/msvd/annotations.json')
    # Output
    parser.add_argument('--feat_h5', type=str, default='/content/Video_Captioning.pytorch/output/metadata')
    args = parser.parse_args()
    pd.set_trace()
    msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
    args.msrvtt_train_range = (0, 6512)
    args.msrvtt_val_range = (6513, 6513 + 497 - 1)
    args.msrvtt_test_range = (6513 + 497, 6513 + 497 + 2990 - 1)
    # msvd_video_sort_lambda = lambda x: int(x[3:-4])
    msvd_video_sort_lambda = lambda x: int(x[5:-4])
    args.msvd_train_range = (0, 1200)
    args.msvd_val_range = (1200, 1200 + 100)
    args.msvd_test_range = (1300, 1300 + 470 - 1)
    args.video_root = args.msrvtt_video_root if args.dataset=='msrvtt' else args.msvd_video_root
    args.video_sort_lambda = msrvtt_video_sort_lambda if args.dataset == 'msrvtt' else msvd_video_sort_lambda
    args.anno_json_path = args.msrvtt_anno_json_path if args.dataset == 'msrvtt' else args.msvd_anno_json_path
    args.train_range = args.msrvtt_train_range if args.dataset == 'msrvtt' else args.msvd_train_range
    args.val_range = args.msrvtt_val_range if args.dataset == 'msrvtt' else args.msvd_val_range
    args.test_range = args.msrvtt_test_range if args.dataset == 'msrvtt' else args.msvd_test_range
    return args

def _sample_frames(opt, video_path, train=True):
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

def sample_frames(opt, video_path, train=True):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frame_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1

    indices = np.linspace(0, frame_count, opt.max_frames, endpoint=False, dtype=int)
    frame_list = np.array(frame_list)[indices]
    return frame_list, frame_count

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

def preprocess_frame_c3d(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
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
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, opt.num_frames, opt.feat_size), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in range(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                frame_list,  frame_count = sample_frames(opt, video_path, train=True)
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

                dataset_feats[n] = feats
                dataset_lens[n] = frame_count

def extract_c3d_features_v1(opt):
    import skvideo.io
    from misc.c3d_keras import C3D_Keras
    from keras.models import Model
    from misc.sports1M_utils import preprocess_input, decode_predictions

    base_model = C3D_Keras(weights='sports1M')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    feature_dim = 4096
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '_v1.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                vid_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                vid = skvideo.io.vread(vid_path)
                indices = np.linspace(0, vid.shape[0], opt.num_frames, endpoint=False, dtype=int)
                frame_list = np.array(vid)[indices]
                x = preprocess_input(frame_list)
                features = model.predict(x)
                dataset_feats[n] = features
                dataset_lens[n] = opt.max_frames

def extract_c3d_features_v2(opt):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    from misc.c3d_keras import C3D_Keras
    from keras.models import Model
    from misc.sports1M_utils import preprocess_input, decode_predictions

    base_model = C3D_Keras(weights='sports1M')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)
    feature_dim = 4096
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '_v2.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                frame_list, frame_count = sample_frames(opt, video_path, train=True)
                clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in frame_list])
                clip = clip[:, :, 44:44 + 112, :]  # crop centrally
                clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
                clip = np.expand_dims(clip, axis=0)  # batch axis
                clip = np.float32(clip)
                #feats = np.zeros((opt.max_frames, opt.feat_size), dtype='float32')
                #frame_list = np.array([preprocess_frame_c3d(x, target_width=112, target_height=112) for x in frame_list])
                #frame_list = Variable(torch.from_numpy(np.float32(frame_list.transpose(3, 0, 1, 2))).unsqueeze(0), volatile=True).cuda()
                features = model.predict(np.transpose(clip, (0, 2, 3, 4, 1)))
                dataset_feats[n] = features
                dataset_lens[n] = opt.max_frames

def extract_c3d_features_v3(opt):
    import skvideo.io
    from misc.c3d_keras import C3D_Keras
    from keras.models import Model
    from misc.sports1M_utils import preprocess_input, decode_predictions

    base_model = C3D_Keras(weights='sports1M')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    feature_dim = 4096
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '_v3.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                vid_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                vid = skvideo.io.vread(vid_path)
                window_size = 16
                clip_feats = []
                for k in xrange(int(len(vid)/window_size)):
                    index = k*16 + 8
                    if index > len(vid):
                        break
                    clip = vid[index - 8: index + 8]
                    x = preprocess_input(clip)
                    features = model.predict(x)
                    clip_feats.append(features)
                dataset_feats[n] = np.asarray(clip_feats).mean(0)
                dataset_lens[n] = opt.max_frames

def extract_p3d_features_v3(opt):
    import skvideo.io
    # from misc.p3d.p3d_model import *
    model = P3D199(pretrained=True, modality='RGB').cuda()
    model.eval()
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    feature_dim = 2048
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '_v3.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                vid_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                vid = skvideo.io.vread(vid_path)
                window_size = 16
                clip_feats = []
                clip_frame = []
                for k in xrange(int(len(vid)/window_size)):
                    index = k*16 + 8
                    if index > len(vid):
                        break
                    clip = vid[index - 8: index + 8]
                    x = np.array([resize(frame, output_shape=(160, 160), preserve_range=True) for frame in clip])
                    clip_frame.append(x)
                data = Variable(torch.from_numpy(np.transpose(np.asarray(clip_frame), (0, 4, 1, 2, 3))), volatile=True).cuda()
                out, fc = model(data.float())
                dataset_feats[n] = (fc.mean(0).data).cpu().numpy()
                dataset_lens[n] = opt.max_frames

def __extract_c3d_features(opt):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    net = C3D(487)
    net.load_state_dict(torch.load('misc/encoder/c3d.pickle'))
    net.cuda()
    net.eval()
    feature_dim = 4096
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                frame_list, frame_count = sample_frames(opt, video_path, train=True)
                frame_list = np.array([preprocess_frame_c3d(x, target_width=112, target_height=112) for x in frame_list])
                frame_list = Variable(torch.from_numpy(np.float32(frame_list.transpose(3, 0, 1, 2))).unsqueeze(0), volatile=True).cuda()
                _, batch_output = net(frame_list, 6)
                dataset_feats[n] = (batch_output.data).cpu().numpy()
                dataset_lens[n] = opt.max_frames

def _extract_c3d_features(opt):
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    net = C3D(487)
    net.load_state_dict(torch.load('misc/encoder/c3d.pickle'))
    net.cuda()
    net.eval()
    feature_dim = 4096
    opt.max_frames = 16
    # Create hdf5 file to save video frame features
    keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
    for i in range(len(keys)):
        h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '.h5'
        if os.path.exists(h5_path): os.remove(h5_path)
        h5 = h5py.File(h5_path, 'w')
        dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, feature_dim), dtype='float32')
        dataset_lens = h5.create_dataset('lens', (values[i][1] - values[i][0] + 1,), dtype='int')
        with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
            for n in xrange(values[i][1] - values[i][0] + 1):
                pbar.update(1)
                video_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                frame_list, frame_count = sample_frames(opt, video_path, train=True)
                clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in frame_list])
                clip = clip[:, :, 44:44 + 112, :]  # crop centrally
                clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
                clip = np.expand_dims(clip, axis=0)  # batch axis
                clip = np.float32(clip)
                #feats = np.zeros((opt.max_frames, opt.feat_size), dtype='float32')
                #frame_list = np.array([preprocess_frame_c3d(x, target_width=112, target_height=112) for x in frame_list])
                #frame_list = Variable(torch.from_numpy(np.float32(frame_list.transpose(3, 0, 1, 2))).unsqueeze(0), volatile=True).cuda()
                _, batch_output = net(Variable(torch.from_numpy(clip)).cuda(), 7)

                dataset_feats[n] = (batch_output.data).cpu().numpy()
                dataset_lens[n] = opt.max_frames

def extract_audio_features(opt):
    import subprocess
    from misc.VGGish import vggish_input
    from misc.VGGish import vggish_params
    from misc.VGGish import vggish_postprocess
    from misc.VGGish import vggish_slim
    import tensorflow as tf

    # Read video list, and sort the videos according to ID
    # Read video list, and sort the videos according to ID
    videos = sorted(os.listdir(opt.video_root), key=opt.video_sort_lambda)
    nvideos = len(videos)
    # Create hdf5 file to save video frame features

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'misc/VGGish/vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        # Create hdf5 file to save video frame features
        keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]
        for i in range(len(keys)):
            h5_path = opt.feat_h5 + '_' + keys[i] + '_' + opt.type + '.h5'
            if os.path.exists(h5_path): os.remove(h5_path)
            h5 = h5py.File(h5_path, 'w')
            dataset_feats = h5.create_dataset('feats', (values[i][1] - values[i][0] + 1, opt.feat_size), dtype='float32')
            #dataset_lens = h5.create_dataset('lens', (nvideos,), dtype='int')
            with tqdm(total=values[i][1] - values[i][0] + 1) as pbar:
                for n in xrange(values[i][1] - values[i][0] + 1):
                    pbar.update(1)
                    video_path = os.path.join(opt.video_root, 'video' + str(n + values[i][0]) + '.mp4')
                    if os.path.exists("tmp.wav"): os.remove("tmp.wav")
                    cmd = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn tmp.wav"
                    try:
                        p2 = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                        p2.wait()
                        examples_batch = vggish_input.wavfile_to_examples("tmp.wav")
                        pproc = vggish_postprocess.Postprocessor('misc/VGGish/vggish_pca_params.npz')
                        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})
                        embedding_batch = embedding_batch.mean(0)
                        dataset_feats[n] = embedding_batch
                    except:
                        print("No audio in file " + video_path)


if __name__ == '__main__':
    opt = parse_opt()
    # pdb.set_trace()

    #build_msvd_annotation(opt)
    #build_msrvtt_videos()

    if opt.type == 'resnet':
        extract_resnet_features(opt, AppearanceEncoder(opt).eval().cuda())
    elif opt.type == 'c3d':
        extract_c3d_features_v3(opt)
    elif opt.type == 'p3d':
        extract_p3d_features_v3(opt)
    elif opt.type == 'audio':
        extract_audio_features(opt)
    else:
        print("You need select one type!")