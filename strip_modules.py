#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import argparse

from fairseq import utils
from fairseq.checkpoint_utils import torch_persistent_save, load_checkpoint_to_cpu


def _strip_params(state, strip_what):
    new_state = state
    new_state['model'] = OrderedDict(
        {key: value for key, value in state['model'].items() if not key.startswith(strip_what)})

    return new_state


def save_state(state, filename):
    torch_persistent_save(state, filename)


def main(args):
    utils.import_user_module(args)
    model_state = load_checkpoint_to_cpu(args.model_path)
    print("Loaded model {}".format(args.model_path))
    model_state = _strip_params(model_state, strip_what=args.strip_what)
    print("Stripped {}".format(args.strip_what))
    save_state(model_state, args.new_model_path)
    print("Saved to {}".format(args.new_model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')
    parser.add_argument('--model-path', type=str, required=True,
                        help="The path to the model to strip")
    parser.add_argument('--new-model-path', type=str, required=True,
                        help="The name for the stripped model")
    parser.add_argument('--strip-what', type=str, default='decoder',
                        help="Part of the network to strip away.")

    main(parser.parse_args())
