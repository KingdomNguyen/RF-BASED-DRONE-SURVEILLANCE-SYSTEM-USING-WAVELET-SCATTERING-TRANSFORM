# generate_lut.py
import numpy as np
import scipy.io as sio
# Parameters
N = int(2 * 10 * 0.85 * 65536)
t = np.linspace(-10, 10, N)
J = 2
Q1 = 8
Q2 = 1
# Generate scales
scales_1 = [2 ** (k/Q1) for k in range(J*Q1)]
scales_2 = [2 ** (k/Q2) for k in range(J*Q2)]
# Generate Gaussian LUT (sigma = 0.7)
sigma_g = 0.7
phi = np.exp(-t**2 / (2 * sigma_g**2))
phi = phi / np.sum(phi)
# Generate Morlet Wavelet LUTs
psi1_lut = []
for s in scales_1:
 sigma = 0.85
 xi = 3 * np.pi / 4
 ts = t / s
 psi = (np.exp(1j * xi * ts) - np.exp(-0.5 * (xi * sigma)**2)) *np.exp(-ts**2 / (2 * sigma**2))
 psi1_lut.append(psi / np.linalg.norm(psi))
psi2_lut = []
for s in scales_2:
 sigma = 0.85
 xi = 3 * np.pi / 4
 ts = t / s
 psi = (np.exp(1j * xi * ts) - np.exp(-0.5 * (xi * sigma)**2)) *
np.exp(-ts**2 / (2 * sigma**2))
 psi2_lut.append(psi / np.linalg.norm(psi))
# Save to MAT file for MATLAB
sio.savemat('D:/wavelet_lut_2.mat', {
 'phi': phi,
 'psi1_lut': np.array(psi1_lut),
 'psi2_lut': np.array(psi2_lut),
 'scales_1': scales_1,
 'scales_2': scales_2,
 't': t
})
