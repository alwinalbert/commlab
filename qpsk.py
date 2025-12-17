import numpy as np
import matplotlib.pyplot as plt

N = int(input("enter number of bits : "))
OF = int(input("enter oversampling factor: "))
FC = int(input("enter carrier frequency (Hz): "))
FS = OF*FC

# Generate random bits
bits = np.random.randint(0, 2, N)
data = np.repeat(bits, OF)

odd = bits[0::2]
even = bits[1::2]

# Nrz encoding
odd_nrz = 2*odd - 1
even_nrz = 2*even - 1

#upsampling
odd_upsampled = np.repeat(odd_nrz, OF)
even_upsampled = np.repeat(even_nrz, OF)

#time axis
t = np.arange(len(odd_upsampled)) / FS
#carrier signals
carrier_cos = np.cos(2 * np.pi * FC * t)
carrier_sin = np.sin(2 * np.pi * FC * t)
#modulated signals
bpsk1 = odd_upsampled * carrier_cos
bpsk2 = even_upsampled * carrier_sin
qpsk_signal = bpsk1 + bpsk2
# Plotting

plt.figure(figsize = (12,8))
#qpsk signal
plt.subplot(4,1,1)
plt.title("qpsk time domain signal")
plt.plot(t, qpsk_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
# data
plt.subplot(4,1,2)
plt.title("data")
plt.plot(data)
plt.xlabel("Bit Index")
plt.ylabel("Bit Value")
plt.grid()
#upsampled data
plt.subplot(4,1,3)
plt.title("odd bits after NRZ encoding and upsampling")
plt.plot(odd_upsampled)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
#constellation diagram
plt.subplot(4,1,4)
plt.title("constellation diagram")
plt.scatter(odd_nrz, even_nrz)
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.grid()
plt.tight_layout()
plt.show()