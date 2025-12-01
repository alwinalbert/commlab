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
noise = np.random.randn(len(bpsk))
bpsk_noisy = bpsk + noise
    
detection = bpsk_noisy*carrier

detected_symbols = []
for i in range(len(bits)):
    start = i * samples_per_bit
    end = (i+1) * samples_per_bit
    
    # Integrate products for one bit
    s = np.sum(detection[start:end])
    
    #  Decision rule
    if s >= 0:
        detected_symbols.append(1)
    else:
        detected_symbols.append(0)

detected_symbols = np.array(detected_symbols)

plt.figure(figsize=(10, 8))

plt.subplot(6,1,1)
plt.title("input bits")
plt.plot(t, np.repeat(bits, samples_per_bit), drawstyle='steps-post')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,2)
plt.title("carrier wave (normal cosine wave)")
plt.plot(t, carrier)
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,3)
plt.title("symbol wave")
plt.plot(t, symbol_wave)
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,4)
plt.title("BPSK modulated signal")
plt.plot(t, bpsk)
plt.xlabel("Time (s)")
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,5)
plt.title("BPSK received signal with noise")            
plt.plot(t, bpsk_noisy)
plt.xlabel("Time (s)")
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(6,1,6)
plt.title("input signal")
plt.plot(t, detected_symbols.repeat(samples_per_bit), drawstyle='steps-post')
plt.grid(True)
plt.tight_layout()
plt.show()