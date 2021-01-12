import logging
import os
import sys

import h5py
import numpy as np
import torch

from fairseq.data.indexed_dataset import IndexedDatasetBuilder
from fairseq.options import get_parser


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq.examples.preprocess_audio')


def reader_h5(path):
    with h5py.File(path, "r") as file:
        key_list = list(file.keys())
        key_list.sort(key=lambda x: int(x))
        for key in key_list:
            yield torch.from_numpy(file[str(key)].value)


def reader_npz(path):
    with open(path, 'rb') as f:
        shape = np.load(f)
        for i in range(int(shape[0])):
            yield torch.from_numpy(np.load(f))


SUPPORTED_TYPES = {'h5': reader_h5, 'npz': reader_npz}


def get_reader(fmt):
    return SUPPORTED_TYPES[fmt]


class AudioIndexedDatasetBuilder(IndexedDatasetBuilder):
    def __init__(self, out_file, fix_lua_indexing=False):
        super().__init__(out_file, dtype=np.float32)
        self.fix_lua_indexing = fix_lua_indexing

    def add_item(self, tensor):
        np_tensor = tensor.numpy()
        # +1 for Lua compatibility
        if self.fix_lua_indexing:
            np_tensor += 1
        bytes = self.out_file.write(np.array(np_tensor, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))


def main(args):
    def make_dataset(input_prefix, output_prefix):
        dest_file_prefix = os.path.join(args.destdir, output_prefix + ".npz")
        ds = AudioIndexedDatasetBuilder(
            "{}.{}".format(dest_file_prefix, 'bin'), fix_lua_indexing=args.legacy_audio_fix_lua_indexing)

        def consumer(tensor):
            ds.add_item(tensor)

        def binarize(input_file, audio_reader, consumer):
            nseq, nsamp = 0, 0
            for tensor in audio_reader(input_file):
                consumer(tensor)
                nseq += 1
                nsamp += tensor.size(0)
            return {'nseq': nseq, 'nsamp': nsamp}

        input_file = '{}.{}'.format(input_prefix, args.format)
        audio_reader = get_reader(args.format)
        res = binarize(input_file, audio_reader, consumer)
        logger.info('| [{}] {}: {} audio_seq, {} audio_samples'.format(
            args.format, input_file, res['nseq'], res['nsamp']))
        ds.finalize("{}.{}".format(dest_file_prefix, 'idx'))

    if args.trainpref:
        make_dataset(args.trainpref, "train")
    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            make_dataset(validpref, outprefix)
    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            make_dataset(testpref, outprefix)


def get_preprocessing_parser():
    parser = get_parser("Preprocessing", "translation")
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    parser.add_argument('--format', metavar='INP',
                        help='Input format for audio files')
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes")
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")
    parser.add_argument('--legacy-audio-fix-lua-indexing', action='store_true', default=False,
                        help='if set, the input filterbanks are added 1 for compatibility with lua indexing fix')
    # TODO: add parallel implementation
    # fmt: on
    return parser


if __name__ == "__main__":
    parser = get_preprocessing_parser()
    args = parser.parse_args()
    main(args)
