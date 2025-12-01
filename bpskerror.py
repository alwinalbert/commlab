import numpy as np
import matplotlib.pyplot as plt

N = 1_000_000
Eb_N0_dB = np.arange(0, 11, 1)
ber = []
 
for eb_n0_db in Eb_N0_dB:
    eb_n0 = 10**(eb_n0_db/10)
    bits = np.random.randint(0, 2, N)
    x = 2*bits - 1 

    noise_std = 1/np.sqrt(2*eb_n0)
    noise = noise_std * np.random.randn(N)
    received = x + noise

    detected_bits = np.zeros(N)
    for i in range(N):
        if received[i] >= 0:
            detected_bits[i] = 1
        else:
            detected_bits[i] = 0
    
    errors = np.sum(bits != detected_bits)
    ber_value = errors / N
    ber.append(ber_value)
    print(f"Eb/N0 = {eb_n0_db} dB | Errors = {errors} | BER = {ber_value}")

plt.semilogy(Eb_N0_dB, ber, 'o-')
plt.title("BPSK Bit Error Rate vs Eb/N0")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which='both')
plt.show()
