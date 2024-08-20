# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:10:46 2024

@author: sergiozc

@references:
    - https://www.zylia.co/white-paper.html
    - Higuchi, etal. Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR.
"""

import numpy as np
from utils import utils
import matplotlib.pyplot as plt
from NTT_MVDR import NTT_BF
import soundfile as sf

plt.close('all')

# DATA
nc = 19     # Number of channels
# Convert signal to adc (compressed)
y, fs = utils.wav2adc(nc, 'input_data/zylia/signal_{}.wav')
y = utils.interp_if_zeros(y)


# Optimal Beamformer
R_x, enh = NTT_BF.ntt_bf(y, nc)

# Normalized signal
normalized_enh = enh / np.max(np.abs(enh))
# Save the signal as a WAV file
sf.write('signal_after.wav', normalized_enh, fs)


# Save original signal as a WAV
# central_channel = y[:, (nc+1)//2 - 1]
# central_channel = central_channel / np.max(np.abs(central_channel))
# wav.write('results/signal_before.wav', fs, central_channel.astype(np.float32))
