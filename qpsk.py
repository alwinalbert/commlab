import numpy as np
import matplotlib.pyplot as plt

fc = float(input("Enter carrier frequency (Hz): "))
fs = float(input("Enter sampling frequency (Hz): "))
rb = float(input("Enter bit rate (bits per second): "))

tb = 1/rb

# generate bits
bits = np.random.randint(0,2,20)

# split bits for I and Q
I_bits = bits[0::2]
Q_bits = bits[1::2]

# NRZ mapping
I_symbols = 2*I_bits - 1
Q_symbols = 2*Q_bits - 1

# time axis
t = np.arange(0, tb*len(I_symbols), 1/fs)

samples_per_symbol = int(fs*tb)

# upsample
I_wave = np.repeat(I_symbols, samples_per_symbol)
Q_wave = np.repeat(Q_symbols, samples_per_symbol)

# carriers
carrier_cos = np.cos(2*np.pi*fc*t)
carrier_sin = np.sin(2*np.pi*fc*t)

# modulation
qpsk = I_wave*carrier_cos + Q_wave*carrier_sin

# plotting
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.title("Input bits")
plt.plot(np.repeat(bits, samples_per_symbol), drawstyle='steps-post')
plt.grid(True)

plt.subplot(4,1,2)
plt.title("I symbol wave")
plt.plot(t, I_wave)
plt.grid(True)

plt.subplot(4,1,3)
plt.title("Q symbol wave")
plt.plot(t, Q_wave)
plt.grid(True)

plt.subplot(4,1,4)
plt.title("QPSK Modulated Signal")
plt.plot(t, qpsk)
plt.grid(True)

plt.tight_layout()
plt.show()

# constellation
plt.figure()
#plt.scatter(I_symbols, Q_symbols)
plt.scatter(I_symbols,np.zeros(len(I_symbols)))
plt.scatter(np.zeros(len(Q_symbols)),Q_symbols)
plt.axhline(0)
plt.axvline(0)
plt.title("QPSK Constellation")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()