import numpy as np
import matplotlib.pyplot as plt

fc = float(input("Enter carrier frequency (Hz): "))
fs = float(input("Enter sampling frequency (Hz): "))
rb = float(input("Enter bit rate (bits per second): "))
tb = 1/rb

bits = np.random.randint(0,2,20)
symbols = 2*bits - 1

t = np.arange(0,tb*len(bits),1/fs)
carrier = np.cos(2*np.pi*fc*t)

samples_per_bit = int(fs*tb)
symbol_wave = np.repeat(symbols, samples_per_bit)
bpsk = symbol_wave*carrier
plt.figure(figsize=(12, 10))

plt.subplot(4,1,1)
plt.title("input bits")
plt.plot(t, np.repeat(bits, samples_per_bit), drawstyle='steps-post')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,1,2)
plt.title("carrier wave (normal cosine wave)")
plt.plot(t, carrier)
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,1,3)
plt.title("symbol wave")
plt.plot(t, symbol_wave)
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4,1,4)
plt.title("BPSK modulated signal")
plt.plot(t, bpsk)
plt.xlabel("Time (s)")
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()