import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

N = int(input("enter number of bits : "))
OF = int(input("enter oversampling factor: "))
FC = int(input("enter carrier frequency (Hz): "))
FS = OF*FC

# Generate random bits
bits = np.random.randint(0, 2, N)

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

# ebn0 range
ebnodb = np.arange(-20,11,2)
ber = np.zeros(len(ebnodb))
ber_theory = np.zeros(len(ebnodb))

for i in range(len(ebnodb)):
    #noisepower
    ebn0 = 10**(ebnodb[i]/10)
    noise_std = 1/np.sqrt(2*ebn0)

    # Generate noise
    noise_odd = noise_std * np.random.randn(len(bpsk1))
    noise_even = noise_std * np.random.randn(len(bpsk2))
    #received signals
    r_odd = bpsk1 + noise_odd
    r_even = bpsk2 + noise_even
    #demodulation
    r_odd_demod = r_odd * carrier_cos
    r_even_demod = r_even * carrier_sin
    #integration
    r_odd_samples = np.convolve(r_odd_demod, np.ones(OF))[OF-1::OF]
    r_even_samples = np.convolve(r_even_demod, np.ones(OF))[OF-1::OF]
    #decision
    r_odd_bits = r_odd_samples > 0
    r_even_bits = r_even_samples > 0
    #count error 
    errors = np.sum(r_odd_bits != odd) + np.sum(r_even_bits != even)
    ber[i] = errors / N
    # theoretical ber
    ber_theory[i] = 0.5 * erfc(np.sqrt(ebn0))

plt.semilogy(ebnodb, ber,label='Simulated BER')
plt.semilogy(ebnodb, ber_theory,label='Theoretical BER')
plt.title('QPSK Bit Error Rate')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid()
plt.legend()
plt.show()



