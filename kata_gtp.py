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

"""GTP-compliant entry point for Minigo."""

import os
import sys
import json

from absl import app, flags

from gtp_cmd_handlers import (
    BasicCmdHandler, KgsCmdHandler, GoGuiCmdHandler, MiniguiBasicCmdHandler, RegressionsCmdHandler)
import gtp_engine
from kata_strategies import MCTSPlayer, CGOSPlayer
from utils import dbg
import tensorflow as tf


flags.DEFINE_bool('cgos_mode', False, 'Whether to use CGOS settings.')

flags.DEFINE_bool('kgs_mode', False, 'Whether to use KGS courtesy-pass.')

flags.DEFINE_bool('minigui_mode', False, 'Whether to add minigui logging.')

flags.DEFINE_string('save_dir', None, 'Path to model save files.')

flags.DEFINE_string('name_scope', None, 'Name scope of model file.')



# See mcts.py, strategies.py for other configurations around gameplay

FLAGS = flags.FLAGS


def make_gtp_instance(save_dir, name_scope, cgos_mode=False, kgs_mode=False,
                      minigui_mode=False):
    """Takes a path to model files and set up a GTP engine instance."""
    # Here so we dont try load EdgeTPU python library unless we need to

    load_file = os.path.join(save_dir,"variables","variables")
    model_config_json = os.path.join(save_dir,"model.config.json")

    with open(model_config_json) as f:
        model_config = json.load(f)

    if load_file.endswith(".tflite"):
        from dual_net_edge_tpu import DualNetworkEdgeTpu
        n = DualNetworkEdgeTpu(load_file)
    else:
        from kata_net import Model
        with tf.compat.v1.variable_scope(name_scope):
            n = Model(load_file, model_config, 19, {})

    if cgos_mode:
        player = CGOSPlayer(network=n, seconds_per_move=5, timed_match=True,
                            two_player_mode=True)
    else:
        player = MCTSPlayer(network=n, two_player_mode=True)

    name = "Minigo-" + os.path.basename(load_file)
    version = "0.2"

    engine = gtp_engine.Engine()
    engine.add_cmd_handler(
        gtp_engine.EngineCmdHandler(engine, name, version))

    if kgs_mode:
        engine.add_cmd_handler(KgsCmdHandler(player))
    engine.add_cmd_handler(RegressionsCmdHandler(player))
    engine.add_cmd_handler(GoGuiCmdHandler(player))
    if minigui_mode:
        engine.add_cmd_handler(MiniguiBasicCmdHandler(player, courtesy_pass=kgs_mode))
    else:
        engine.add_cmd_handler(BasicCmdHandler(player, courtesy_pass=kgs_mode))

    return engine


def main(argv):
    """Run Minigo in GTP mode."""
    del argv
    engine = make_gtp_instance(FLAGS.save_dir,
                                FLAGS.name_scope,
                               cgos_mode=FLAGS.cgos_mode,
                               kgs_mode=FLAGS.kgs_mode,
                               minigui_mode=FLAGS.minigui_mode)
    dbg("GTP engine ready\n")
    for msg in sys.stdin:
        if not engine.handle_msg(msg.strip()):
            break


if __name__ == '__main__':
    app.run(main)
