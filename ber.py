import numpy as np
import matplotlib.pyplot as plt

bits = np.array([int(b) for b in input("Enter the binary sequence (e.g., 1101): ")])
bits = np.tile(bits,10000)
N = len(bits)
ber = []
ebn0 = np.arange(0,11)
for snrdb in ebn0:
    snr = 10**(snrdb/10)
    noise_std =  np.sqrt(1/(2*snr))
    x = 2*bits-1
    noise = noise_std*np.random.randn(N)
    r = x+noise
    detected = (r>=0).astype(int)
    errors = np.sum(bits != detected)
    ber.append(errors/N)

plt.figure(figsize=(8,6))
plt.semilogy(ebn0, ber)
plt.xlabel('E_b/N_0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate for BPSK in AWGN')
plt.grid(True)  
plt.show()