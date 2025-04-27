import torch
import torchaudio
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

def ReLU(x):
    return x * (x > 0)
def save_audio(waveform, sample_rate, audio_name = "audio", n_fft = 256, eps = 1e-10):
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
    fig, axes = plt.subplots(2, 1, figsize=(16,8))
    timesteps = torch.arange(waveform.shape[-1]) / sample_rate
    spectrogram = transform(waveform)
    axes[0].plot(timesteps, waveform)
    axes[1].imshow(10*torch.log10(spectrogram + eps), origin="lower")

    fig.savefig(f'{audio_name}.png')
    torchaudio.save(f'{audio_name}.wav', waveform.unsqueeze(0), sample_rate)

@torch.no_grad()
def estimate_fir_lstsq(x, y, num_taps, normalization_factor=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = len(x)
    X = torch.zeros((N, num_taps))
    for i in range(num_taps):
        X[:, i] = torch.roll(x, i)
    X[:num_taps-1, :] = 0  # zero out the wrapped part
    h, _, _, _ = torch.linalg.lstsq(X.to(device), y.to(device), rcond=None)
    return h.cpu()

@torch.no_grad()
def estimate_fir_rls(x, y, num_taps, lam=0.99, delta=1e3, normalization_factor=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = len(x)
    h = torch.zeros(num_taps).to(device)
    P = delta * torch.eye(num_taps).to(device)
    x_reverse = torch.flip(x, dims=[0]).to(device) * normalization_factor
    for n in range(num_taps, N):
        x_n = x_reverse[N-n: N-n+num_taps]  # Take x[n], x[n-1], ..., x[n-M+1]
        if len(x_n) < num_taps:
            # pad with zeros if at the beginning
            x_n = torch.pad(x_n, (0, num_taps - len(x_n)))

        Pi_x = P @ x_n
        k = Pi_x / (lam + x_n @ Pi_x)
        e = y[n] - h @ x_n
        h = h + k * e
        P = (P - torch.outer(k, x_n) @ P) / lam

        if n % int(N/100) == 0:
            print(f"updated step {n} out of {N}")

    return h

# alternative: FIR
def estimate_fir_gd(x, y, num_taps, lr=1e-2, num_steps=10000, len_training_seg_prop=1.0, normalization_factor=0.01):
    """
    Estimate FIR filter h from input x and output y using gradient descent.
    Args:
        x: Input signal, 1D torch tensor
        y: Output signal, 1D torch tensor
        num_taps: Number of FIR filter taps
        lr: Learning rate
        num_steps: Number of gradient steps
    Returns:
        h: Estimated FIR filter, 1D torch tensor
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    M = y.shape[0]
    len_training_seg = int(num_taps * len_training_seg_prop)
    assert M > len_training_seg
    assert len_training_seg >= num_taps
    
    # Zero pad x
    x_padded = torch.nn.functional.pad(x, (num_taps-1, 0))
    
    # Build Toeplitz matrix
    X = torch.zeros((M, num_taps), dtype=x.dtype)
    for i in range(num_taps):
        X[:, i] = x_padded[num_taps-1-i:M+num_taps-1-i]
    
    # Initialize h as learnable parameter
    X = X.to(device) * normalization_factor
    y = y.to(device)
    h = torch.nn.Parameter(torch.randn(num_taps, requires_grad=True)).to(device)
    
    optimizer = torch.optim.Adam([h], lr=lr)
    loss_fn = torch.nn.MSELoss()

    for step in range(num_steps):
        optimizer.zero_grad()
        time_start = torch.randint(0, M - len_training_seg, ())
        y_pred = X[time_start:time_start+len_training_seg] @ h
        loss = loss_fn(y_pred, y[time_start:time_start+len_training_seg])
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    return h.detach()

def fir_filter(x, h, normalization_factor = 0.01):
    """
    Apply FIR filter to input x with coefficients h.
    Args:
        x: (batch_size, 1, signal_length) torch tensor
        h: (num_taps,) torch tensor
    Returns:
        y: (batch_size, 1, output_length) torch tensor
    """
    # Flip h for convolution
    h_flipped = h.flip(0).unsqueeze(0).unsqueeze(0)  # (out_channels, in_channels, kernel_size)
    
    # Apply 1D convolution
    y = F.conv1d(x * normalization_factor, h_flipped, padding=0)
    return y