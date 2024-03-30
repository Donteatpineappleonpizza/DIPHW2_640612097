import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt
# Corrected code snippet with cv2.imread used to read the PGM image for both operations

# We'll perform two operations:
# 1. Inverse Fourier transform using only the phase information (image with no amplitude).
# 2. Inverse Fourier transform using only the amplitude information (image with no phase).

# Read the PGM image directly using cv2.imread
image_path = 'Lenna.pgm'
ima = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Pad the image to 256x256 (closest 2^n)
pad_size = (256 - ima.shape[0]) // 2
padimage = np.pad(ima, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

# Compute the Fourier transform
imagefft = fft2(np.float32(padimage))

# Set the amplitude data to one for the phase-only image
shiftedfft_one_amp = np.exp(1j * np.angle(imagefft))

# Inverse shift the Fourier transform for the phase-only image
shift_idft_one_amp = ifftshift(shiftedfft_one_amp)

# Inverse Fourier transform for the phase-only image
image_one_amp = ifft2(shift_idft_one_amp)

# Take the real part of the image for the phase-only image
image_one_amp = np.real(image_one_amp)

# Display the phase-only result
plt.figure(figsize=(6, 6))
plt.imshow(image_one_amp, cmap='gray')
plt.title('Image with no amplitude')
plt.axis('on')
plt.show()

# Set the phase to zero for the amplitude-only image
shiftedfft_zero_phase = np.abs(imagefft)

# Inverse shift the Fourier transform for the amplitude-only image
shift_idft_zero_phase = ifftshift(shiftedfft_zero_phase)

# Inverse Fourier transform for the amplitude-only image
image_zero_phase = ifft2(shift_idft_zero_phase)

# Take the real part of the image for the amplitude-only image
image_zero_phase = np.real(image_zero_phase)

# Display the amplitude-only result
plt.figure(figsize=(6, 6))
plt.imshow(image_zero_phase, cmap='gray')
plt.title('Image with no phase')
plt.axis('on')
plt.show()
