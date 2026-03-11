import numpy as np
import matplotlib.pyplot as plt

fc = float(input("enter ccarrier frequency: "))
fs = float(input("enter the sampling frequency: "))
rb = float(input("enter the bit rate:"))
tb = 1/rb

bits = np.random.randint(0,2,20)
symbols = 2*bits-1
t= np.arange(0,tb*len(bits),1/fs)
carrier = np.cos(2*np.pi*fc*t)
carrier2 = np.sin(2*np.pi*fc*t)
samples_per_bit = int(fs*tb)
symbol_wave = np.repeat(symbols,samples_per_bit)
bpsk = symbol_wave*carrier
bpsk2 = symbol_wave*carrier2
plt.figure(figsize=(12,10))
plt.subplot(5,1,1)
plt.title("input bits")
plt.plot(t, np.repeat(bits, samples_per_bit))
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(5,1,2)
plt.title("carrier wave (normal cosine wave)")
plt.plot(t, carrier)
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(5,1,3)
plt.title("symbol wave")
plt.plot(t, symbol_wave)
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(5,1,4)
plt.title("BPSK modulated signal")
plt.plot(t, bpsk)
plt.xlabel("Time (s)")
plt.subplot(5,1,5)
plt.title("detection signal (product of BPSK and sine wave)")
plt.plot(t, bpsk2)
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

I = symbols
Q = symbols
plt.figure(figsize=(6,6))
plt.scatter(I,np.zeros(len(I)))
plt.scatter(np.zeros(len(Q)),Q)
plt.title("Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()