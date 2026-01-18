import numpy as np
import matplotlib.pyplot as plt
T = 1
Fs = 100
rolloff = 0.7

# Raised Cosine pulse
g = lambda t: np.sinc(t) * np.cos(np.pi * rolloff * t) / (1 - (2 * rolloff * t) ** 2)

# Binary message
binary_sequence = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Bipolar conversion
j = np.array(binary_sequence) * 2 - 1

# Time axis
t = np.arange(-2 * T, (len(j) + 2) * T, 1 / Fs)

# Pulse-shaped signal
y = sum(j[k] * g(t - k * T) for k in range(len(j)))

# -----------------------------
# Subplots
# -----------------------------
plt.figure(figsize=(12, 8))

# 1️⃣ Binary Message
plt.subplot(2, 2, 1)
plt.step(np.arange(len(binary_sequence)), binary_sequence)
plt.title("Binary Message")
plt.xlabel("Bit index")
plt.ylabel("Amplitude")
plt.grid(True)

# 2️⃣ Pulse-shaped Signal
plt.subplot(2, 2, 2)
plt.plot(t, y)
plt.title("Pulse Shaped Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# 3️⃣ Raised Cosine Impulse Response
plt.subplot(2, 2, 3)
t_rc = np.arange(-5, 5, 0.01)
plt.plot(t_rc, g(t_rc))
plt.title("Raised Cosine Impulse Response")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# 4️⃣ Eye Diagram
plt.subplot(2, 2, 4)
x = np.arange(-T, T, 1 / Fs)

for i in range(2 * Fs, len(y) - 3 * Fs, Fs):
    plt.plot(x, y[i:i + 2 * Fs], color='green')

plt.title("Eye Diagram")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()