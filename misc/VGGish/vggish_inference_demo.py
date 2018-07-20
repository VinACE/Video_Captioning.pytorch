# Copyright 2017 The TensorFlow Authors All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================


r"""A simple demonstration of running VGGish in inference mode.



This is intended as a toy example that demonstrates how the various building

blocks (feature extraction, model definition and loading, postprocessing) work

together in an inference context.



A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted

into log mel spectrogram examples, fed into VGGish, the raw embedding output is

whitened and quantized, and the postprocessed embeddings are optionally written

in a SequenceExample to a TFRecord file (using the same format as the embedding

features released in AudioSet).



Usage:

  # Run a WAV file through the model and print the embeddings. The model

  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are

  # loaded from vggish_pca_params.npz in the current directory.

  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file



  # Run a WAV file through the model and also write the embeddings to

  # a TFRecord file. The model checkpoint and PCA parameters are explicitly

  # passed in as well.

  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \

                                    --tfrecord_file /path/to/tfrecord/file \

                                    --checkpoint /path/to/model/checkpoint \

                                    --pca_params /path/to/pca/params



  # Run a built-in input (a sine wav) through the model and print the

  # embeddings. Associated model files are read from the current directory.

  $ python vggish_inference_demo.py

"""

from __future__ import print_function
import argparse
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import h5py
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os

flags = tf.app.flags
flags.DEFINE_string('wav_file', None,'Path to a wav file. Should contain signed 16-bit PCM samples. If none is provided, a synthetic sound is used.')
flags.DEFINE_string('checkpoint', 'vggish_model.ckpt','Path to the VGGish checkpoint file.')
flags.DEFINE_string('pca_params', 'vggish_pca_params.npz', 'Path to the VGGish PCA parameters file.')
flags.DEFINE_string('tfrecord_file', None, 'Path to a TFRecord file where embeddings will be written.')
FLAGS = flags.FLAGS

def parse_opt():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--frame_sample_rate', type=int, default=10)  # Video sample rate
    parser.add_argument('--feat_size', type=int, default=128)
    parser.add_argument('--type', type=str, default='mfcc', help="[renset, motion, audio, category, c3d]")
    parser.add_argument('--max_audio_clips', type=int, default=20)  # Number of frames
    # Models
    # parser.add_argument('--vggish_checkpoint', type=str, default='./misc/encoder/pytorch-resnet/resnet101.pth')
    # MSR_VTT
    parser.add_argument('--msrvtt_video_root', type=str, default='./amw/')
    # MSVD
    parser.add_argument('--msvd_video_root', type=str, default='./amw')
    # Output
    parser.add_argument('--feat_h5', type=str, default='./h5s/')
    args = parser.parse_args()

    msrvtt_video_sort_lambda = lambda x: int(x[5:-9])
    args.msrvtt_train_range = (0, 6513 - 1)
    args.msrvtt_val_range = (6512, 6512 + 2990 - 1)
    args.msrvtt_test_range = (6512 + 2990, 6512 + 2990 + 497 - 1)
    # msvd_video_sort_lambda = lambda x: int(x[3:-4])
    msvd_video_sort_lambda = lambda x: int(x[5:-4])
    args.msvd_train_range = (0, 1200 - 1)
    args.msvd_val_range = (1200, 1200 + 100 - 1)
    args.msvd_test_range = (1300, 1300 + 470 - 1)

    args.video_root = args.msrvtt_video_root if args.dataset == 'msrvtt' else args.msvd_video_root
    args.video_sort_lambda = msrvtt_video_sort_lambda if args.dataset == 'msrvtt' else msvd_video_sort_lambda
    args.train_range = args.msrvtt_train_range if args.dataset == 'msrvtt' else args.msvd_train_range
    args.val_range = args.msrvtt_val_range if args.dataset == 'msrvtt' else args.msvd_val_range
    args.test_range = args.msrvtt_test_range if args.dataset == 'msrvtt' else args.msvd_test_range
    return args


# --wav_file=./amw/$line --tfrecord_file=./h5s/$line.h5 --pca_params=vggish_pca_params.npz --checkpoint=vggish_model.ckpt

def main(_):
    opt = parse_opt()
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        keys, values = ['train', 'val', 'test'], [opt.train_range, opt.val_range, opt.test_range]

        for i in range(3):
            h5_path = opt.feat_h5 + '2016' + '_' + keys[i] + '_' + opt.type + '.h5'
            if os.path.exists(h5_path): os.remove(h5_path)
            h5 = h5py.File(h5_path, 'w')
            dataset_feats = h5.create_dataset('feats', ((values[i][1] - values[i][0] + 1), opt.feat_size), dtype='float32')
            # print(values[i])
            for audio_id in range(values[i][0], values[i][1] + 1):
                wav_file = opt.video_root + 'video' + str(audio_id) + '.mp4.wav'
                #print(wav_file)
                # id = int(audio_id[5:-9])
                #print(audio_id)
                if os.path.isfile(wav_file):
                    examples_batch = vggish_input.wavfile_to_examples(wav_file)
                    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
                    writer = tf.python_io.TFRecordWriter(FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None
                    [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})
                    #print(len(embedding_batch), len(embedding_batch[0]))
                    embedding_batch = embedding_batch.mean(0)
                    dataset_feats[audio_id - values[i][0]] = embedding_batch
                    #print(embedding_batch)

    if writer:
        writer.close()

if __name__ == '__main__':
    tf.app.run()
