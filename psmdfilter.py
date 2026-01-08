import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

num_bits = 10
sps = 8                       # Samples per symbol
alpha = 0.4                   # Roll-off factor
span = 6                      # Filter span in symbols
noise_variance = 0.1

# 1. BIT GENERATION
bits = np.random.randint(0, 2, num_bits)

# NRZ encoding
symbols = 2 * bits - 1

# 2. UPSAMPLING (USING np.repeat)
upsampled = np.repeat(symbols, sps)

# 3. RAISED COSINE FILTER
def raised_cosine_filter(alpha, span, sps):
    t = np.arange(-span/2, span/2 + 1/sps, 1/sps)
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0
        elif alpha != 0 and abs(ti) == 1 / (2 * alpha):
            h[i] = (np.pi / 4) * np.sinc(1 / (2 * alpha))
        else:
            h[i] = (np.sinc(ti) *
                    np.cos(np.pi * alpha * ti) /
                    (1 - (2 * alpha * ti) ** 2))

    return h / np.sqrt(np.sum(h**2))

pulse = raised_cosine_filter(alpha, span, sps)

# 4. TRANSMITTER (PULSE SHAPING)
tx_signal = convolve(upsampled, pulse, mode='same')

# 5. CHANNEL (AWGN)
noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# 6. MATCHED FILTER
matched_filter_output = convolve(rx_signal, pulse, mode='same')

# 7. SAMPLING
# With mode='same', the signal is already centered, so we sample at symbol positions
sample_points = np.arange(sps//2, num_bits*sps, sps)
samples = matched_filter_output[sample_points]

# 8. DECISION DEVICE
detected_bits = (samples > 0).astype(int)

# 9. PLOTS
fig = plt.figure(figsize=(14, 10))

# Subplot 1: Raised Cosine Filter Impulse Response
plt.subplot(3, 2, 1)
plt.plot(pulse, 'b-', linewidth=2)
plt.title("Raised Cosine Filter Impulse Response", fontsize=11, fontweight='bold')
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Subplot 2: Transmitted Signal
plt.subplot(3, 2, 2)
plt.plot(tx_signal)
plt.title("Transmitted Signal (Pulse Shaped)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Subplot 3: Received Signal
plt.subplot(3, 2, 3)
plt.plot(rx_signal)
plt.title("Received Signal (with AWGN)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Subplot 4: Matched Filter Output
plt.subplot(3, 2, 4)
plt.plot(matched_filter_output, 'm-', linewidth=1.5, label='Matched Filter Output')
plt.plot(sample_points, samples, 'ro', markersize=8, label='Sample Points')
plt.title("Matched Filter Output")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Original Bits
plt.subplot(3, 2, 5)
plt.stem(bits)
plt.title("Original Bits")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.ylim([-0.5, 1.5])
plt.grid(True, alpha=0.3)

# Subplot 6: Detected Bits
plt.subplot(3, 2, 6)
plt.stem(detected_bits)
plt.title("Detected Bits")
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.ylim([-0.5, 1.5])
plt.grid(True, alpha=0.3)

plt.suptitle("Matched Filter Communication System")
plt.tight_layout()
plt.show()

print("Original bits :", bits)
print("Detected bits :", detected_bits)
