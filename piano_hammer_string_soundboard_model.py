import io
import os

import math
import numpy as np
from scipy import signal
from scipy.signal import buttord, butter

import torch
import torchaudio

from utils import ReLU, save_audio, estimate_fir_gd, fir_filter

# string physics parameters
m_string = 3.93e-3 # kg
L_string = 0.62 # m
density_string = m_string / L_string # kg/m
tension_string = 670 # N
E_string = 2.02e11 # Young's modulus N/m2
r_string = 5e-4 # radius, m
pi = 3.141592653
S_string = pi*r_string*r_string
kappa_string = r_string / 2 # radius of gyration

B = kappa_string**2 * E_string * S_string * (pi/L_string)**2 / tension_string
f0 = math.sqrt(tension_string / density_string) / (2*L_string)
b1 = 0.5 #s^{-1}
b3 = 6.25e-9 # s

# hammer physics parameters
x_hammer = (1-0.12)*L_string #m
m_hammer = 2.97e-3 #kg
stiff_exp = 2.5 #ph
stiff_coef = 4.5e9 #Kh
v_hammer_init = 1.0 # 2.0 #m/s

# simulation settings
fs = 24000 # sample rate, Hz
num_harmonics = 37 # string harmonics
harmonic_indices = torch.LongTensor(np.arange(num_harmonics) + 1)

# simulation coeffs
f_harmonics = f0 * harmonic_indices * torch.sqrt(1 + B * harmonic_indices**2)
# f_harmonics_another_version = 1/(2*pi)*torch.sqrt(a_ode_0 - a_ode_1**2 /4)
inharmonicity = b1 + b3 * 2*pi * f_harmonics # R_k

a0_temp = (harmonic_indices * pi / L_string) ** 2
a_ode_0 = a0_temp * tension_string / density_string + a0_temp**2 * E_string * S_string * kappa_string ** 2 / density_string
a_ode_1 = 2 * inharmonicity
b_ode = 2 / (L_string * density_string)
tau = 1 / inharmonicity
A_impulse_response = b_ode / (4*pi*f_harmonics)

p_dsp = torch.exp(-1 / tau / fs + 1j * 2*pi * f_harmonics / fs)
a_dsp_1 = -2 * torch.real(p_dsp)
a_dsp_2 = torch.abs(p_dsp)**2
b_dsp = A_impulse_response * torch.imag(p_dsp) / fs
w_dsp_in = torch.sin(harmonic_indices * pi * x_hammer / L_string)
w_dsp_out = w_dsp_in
w_bridge_out = tension_string * pi / L_string * harmonic_indices

# simulation loop
max_time = 8 # s
num_iterations = int(max_time * fs)

y_modes = torch.zeros(num_harmonics, num_iterations)
F_hammer_modes = torch.zeros(num_harmonics, num_iterations)
F_hammer = torch.zeros(num_iterations)
y_hammer = torch.zeros(num_iterations)
y_string_at_hammer = torch.zeros(num_iterations)
F_bridge = torch.zeros(num_iterations)

y_hammer[2] = v_hammer_init / fs # treat n=2 as the initial time
y_hammer[0] = -v_hammer_init / fs
for n in range(2, num_iterations - 1):
    delta_y = y_hammer[n] - y_string_at_hammer[n]
    F_hammer[n] = stiff_coef * ReLU(delta_y) ** stiff_exp
    F_hammer_modes[:, n] = w_dsp_in * F_hammer[n]
    y_modes[:, n+1] = b_dsp * F_hammer_modes[:, n] - a_dsp_1 * y_modes[:, n] - a_dsp_2 * y_modes[:, n-1]
    y_string_at_hammer[n+1] = (w_dsp_out * y_modes[:, n+1]).sum()
    y_hammer[n+1] = 2 * y_hammer[n] - y_hammer[n-1] - F_hammer[n] / (m_hammer * fs**2)
    F_bridge[n] = (w_bridge_out * y_modes[:, n]).sum()

save_audio(F_bridge, fs, "F_bridge")

# soundboard model
test_audio_path = "C4.wav"
waveform, sample_rate = torchaudio.load(test_audio_path)
resampler = torchaudio.transforms.Resample(sample_rate, fs, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)[0]

n_samples_to_fit = int(5.9 * fs)
num_taps = 4800 # 20000 # int(fs * 0.5)
input_signal = F_bridge[2:2+n_samples_to_fit]
output_signal = resampled_waveform[:n_samples_to_fit]
normalization_factor = 0.01

soundboard_fir_taps = estimate_fir_gd(input_signal, output_signal, num_taps, len_training_seg_prop=1.0, normalization_factor=normalization_factor)
# soundboard_fir_taps = estimate_fir_rls(input_signal, output_signal, num_taps, normalization_factor=normalization_factor)
# soundboard_fir_taps = estimate_fir_lstsq(input_signal, output_signal, num_taps, normalization_factor=normalization_factor)

soundboard_fir_taps = soundboard_fir_taps.cpu()
waveform_simulated = fir_filter(F_bridge.unsqueeze(0).unsqueeze(0), soundboard_fir_taps).squeeze()
save_audio(waveform_simulated, fs, "waveform_simulated")