import numpy as np
import matplotlib.pyplot as plt
fm = 100
fs =15*fm
A =1
bits= range(0,9)
t = np.arange(0,2/fm,0.0001)
ts = np.arange(0,5/fm,1/fs)
x = A+np.sin(2*np.pi*fm*t)
xs = A+np.sin(2*np.pi*fm*ts)
plt.subplot(2,1,1)
plt.plot(t,x)
plt.subplot(2,1,2)
plt.stem(ts,xs)
plt.show()

L=8
xsmin = np.min(xs)
xsmax = np.max(xs)
delta = (xsmax-xsmin)/L
quantized = np.round((xs-xsmin)/delta)*delta+xsmin
levels = np.array([0,1,2,3,4,5,6,7,6,5,4,3,2,1,0]*10)[:len(xs)]
pcm_bits = [format(i,'03b') for i in levels]
print("PCM Bitstream:")
print(pcm_bits)
plt.stem(ts,quantized)
plt.show()

reconstructed = np.zeros_like(t)
for i in range(len(ts)-1):
    reconstructed[(t>=ts[i])&(t<ts[i+1])]= quantized[i]

reconstructed[t>=ts[-1]] = quantized[-1]

plt.plot(t,reconstructed)
plt.show()

sqnr_values = []
xmax = np.max(x)
xmin= np.min(x)
delta = (xmax-xmin)/2
signalpower = (delta/2)**2 / 2
for b in bits:
    L = 2**b
    delta = (xmax-xmin)/L
    noisepower = (delta**2)/12
    sqnr = 10*np.log10(signalpower/noisepower)
    sqnr_values.append(sqnr)

plt.plot(bits,sqnr_values)
plt.xlabel('Number of bits')
plt.ylabel('SQNR (dB)')
plt.title('SQNR vs Number of Bits')
plt.grid()
plt.show()
