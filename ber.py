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

ebn0db = np.arange(0,11)
berqpsk =[]
for snrdb in ebn0db:
    snr = 10**(snrdb/10)
    i = bits[0::2]
    q = bits[1::2]
    i_nrz = 2*i-1   
    q_nrz = 2*q-1
    noise_std = np.sqrt(1/(2*snr))
    noise_i = noise_std*np.random.randn(N//2)
    noise_q = noise_std*np.random.randn(N//2)
    r_i = i_nrz + noise_i
    r_q = q_nrz + noise_q
    detected_i = (r_i>=0).astype(int)
    detected_q = (r_q>=0).astype(int)
    errors = np.sum(i != detected_i) + np.sum(q != detected_q)
    berqpsk.append(errors/N)
    
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.semilogy(ebn0, ber, label='BPSK')
plt.xlabel('E_b/N_0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate for BPSK in AWGN')
plt.grid(True) 
plt.subplot(2,1,2)
plt.semilogy(ebn0db, berqpsk, label='QPSK')
plt.xlabel('E_b/N_0 (dB)')
plt.ylabel('BER')
plt.title('Bit Error Rate for QPSK in AWGN')
plt.grid(True)
plt.tight_layout()
plt.show()