import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

num_bits = 10
sps = 8
alpha = 0.4
span = 6
noise_variance = 0.1

# 1. BIT GENERATION
bits = np.random.randint(0,2,num_bits)

# NRZ mapping
symbols = 2*bits - 1

# 2. UPSAMPLING
upsampled = np.repeat(symbols, sps)

# 3. RAISED COSINE PULSE
t = np.arange(-span/2, span/2 + 1/sps, 1/sps)

pulse = np.sinc(t) * np.cos(np.pi*alpha*t) / (1 - (2*alpha*t)**2)

# Handle divide-by-zero cases
pulse[np.isinf(pulse)] = 0
pulse[np.isnan(pulse)] = np.pi/4 * np.sinc(1/(2*alpha))

# Normalize filter energy
pulse = pulse / np.sqrt(np.sum(pulse**2))

# 4. TRANSMITTER (Pulse shaping)
tx_signal = convolve(upsampled, pulse, mode='same')

# 5. CHANNEL (AWGN)
noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# 6. MATCHED FILTER
matched_filter_output = convolve(rx_signal, pulse, mode='same')

# 7. SAMPLING
sample_points = np.arange(sps//2, num_bits*sps, sps)
samples = matched_filter_output[sample_points]

# 8. DECISION
detected_bits = (samples > 0).astype(int)

# 9. PLOTS
plt.figure()
plt.title("Raised Cosine Pulse")
plt.plot(t,pulse)
plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.title("Transmitted Signal")
plt.plot(tx_signal)
plt.grid(True)

plt.subplot(2,1,2)
plt.title("Received Signal")
plt.plot(rx_signal)
plt.grid(True)

plt.figure()
plt.title("Matched Filter Output")
plt.plot(matched_filter_output)
plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.stem(bits)
plt.title("Transmitted Bits")

plt.subplot(2,1,2)
plt.stem(detected_bits)
plt.title("Detected Bits")

plt.tight_layout()
plt.show()