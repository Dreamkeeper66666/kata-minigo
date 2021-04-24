# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

from absl import flags
import functools
import json
import logging
import os.path
import struct
import tempfile
import time
import numpy as np
import random
import paddle
from mini_paddle import *
import features as features_lib
import go
import symmetries
import minigo_model 
paddle.disable_static()





flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')

flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


class DualNetwork():
    def __init__(self, save_file):
        self.save_file = save_file
        params = paddle.load(save_file)
        self.pd_model = TFModel()
        self.pd_model.set_dict(params, use_structured_name=False)
        self.pd_model.eval()

    def run(self, position):
        probs, values = self.run_many([position])
        return probs[0], values[0]

    def run_many(self, positions):
        f = get_features()
        processed = [features_lib.extract_features(p, f) for p in positions]
        if FLAGS.use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        probabilities,value = self.pd_model(processed)
        if FLAGS.use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)
        return probabilities, value


def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)





