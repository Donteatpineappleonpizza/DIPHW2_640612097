import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt
# Correcting the padding issue
# Ensuring the image is no larger than 256x256 for padding, otherwise, it's resized to fit.

# Read the PGM image directly using cv2.imread
image_path = 'Cross.pgm'
ima = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Rotate the image by 30 degrees
rows, cols = ima.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
rotated_ima = cv2.warpAffine(ima, M, (cols, rows))

def pad_to_size(image, target_size=256):
    rows, cols = image.shape
    if rows > target_size or cols > target_size:
        # If the image is larger than the target size, resize it down
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        # Calculate padding to center the image within the target size
        pad_vert = (target_size - rows) // 2
        pad_horiz = (target_size - cols) // 2
        image = np.pad(image, ((pad_vert, target_size - rows - pad_vert), 
                               (pad_horiz, target_size - cols - pad_horiz)), 
                       mode='constant', constant_values=0)
    return image

# Pad the rotated image properly
pad_rotated_image = pad_to_size(rotated_ima)

# Compute the Fourier transform of the padded, rotated image
rotated_imagefft = fft2(np.float32(pad_rotated_image))

# Compute the amplitude spectrum and shift it to center
amplitude_spectrum_rotated = np.log1p(np.abs(rotated_imagefft))  # Use log1p for better visualization
amplitude_spectrum_rotated_shifted = fftshift(amplitude_spectrum_rotated)

# Compute the phase spectrum and shift it to center
phase_spectrum_rotated = np.angle(rotated_imagefft)
phase_spectrum_rotated_shifted = fftshift(phase_spectrum_rotated)



# Plotting the original, rotated image alongside its amplitude and phase spectrum
plt.figure(figsize=(10, 5))

# The rotated image
plt.subplot(1, 3, 1)
plt.imshow(rotated_ima, cmap='gray')
plt.title('Rotated Image (30 degrees)')
plt.axis('off')

# Amplitude spectrum of the rotated image
plt.subplot(1, 3, 2)
plt.imshow(amplitude_spectrum_rotated_shifted, cmap='gray')
plt.title('Amplitude spectrum rotated')
plt.axis('off')

# Phase spectrum of the rotated image
plt.subplot(1, 3, 3)
plt.imshow(phase_spectrum_rotated_shifted, cmap='gray')
plt.title('Phase spectrum rotated')
plt.axis('off')

plt.tight_layout()
plt.show()
