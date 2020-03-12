# Copyright 2019 RnD at Spoon Radio
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
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf

In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import torch
import torch.nn as nn
import numpy as np
import random


class SpecAugment(nn.Module):

    def __init__(self, frequency_masking_pars, time_masking_pars,
                 frequency_masking_num, time_masking_num, rate=1.0):
        super().__init__()
        self.frequency_masking_pars = frequency_masking_pars
        self.time_masking_pars = time_masking_pars
        self.frequency_masking_num = frequency_masking_num
        self.time_masking_num = time_masking_num
        self.rate = rate

    def forward(self, batch):
        new_spectrograms = []
        x = batch['net_input']['src_tokens']
        for spectrogram in x:
            if random.random() < self.rate:
                sample = spec_augment(spectrogram, self.frequency_masking_pars,
                                    self.time_masking_pars, self.frequency_masking_num, self.time_masking_num,
                                     )
            else:
                sample = spectrogram
            new_spectrograms += [sample]

        new_spectrograms = torch.stack(new_spectrograms)
        batch['net_input']['src_tokens'] = new_spectrograms
        return batch


def specaugment(mel_spectrogram, frequency_masking_para=27,
                 time_masking_para=100, frequency_masking_num=1, time_masking_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    tau, v = mel_spectrogram.size()

    # Step 1 : Frequency masking (masks can overlap)
    for i in range(frequency_masking_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        mel_spectrogram[:, f0:f0 + f] = 0

    # Step 2 : Time masking (masks can overlap)
    for i in range(time_masking_num):
        t = np.random.uniform(low=1.0, high=min(time_masking_para, tau))
        t = int(t)
        t0 = random.randint(0, tau - t)
        mel_spectrogram[t0:t0 + t, :] = 0

    return mel_spectrogram
