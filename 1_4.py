from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt



# Load the image using skimage.io.imread
image_path = 'Cross.pgm'
original_image = imread(image_path)

# Downsample the image to 100x100 using skimage.transform.resize
# Note: resize function needs the scale in (rows, cols) which matches (height, width)
downsampled_image_sk = resize(original_image, (100, 100), mode='reflect', anti_aliasing=True)

# Compute the 2D Fourier Transform of the downsampled image
ft_image_sk = np.fft.fft2(downsampled_image_sk)
ft_image_shifted_sk = np.fft.fftshift(ft_image_sk)

# Compute amplitude and phase spectra
amplitude_spectrum_sk = np.abs(ft_image_shifted_sk)
phase_spectrum_sk = np.angle(ft_image_shifted_sk)

# Plotting the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original downsampled image using skimage
axes[0].imshow(downsampled_image_sk, cmap='gray')
axes[0].set_title('Downsampled Image (skimage)')
axes[0].axis('on')

# Amplitude Spectrum
axes[1].imshow(np.log1p(amplitude_spectrum_sk), cmap='gray')
axes[1].set_title('Amplitude Spectrum (skimage)')
axes[1].axis('off')

# Phase Spectrum
axes[2].imshow(phase_spectrum_sk, cmap='gray')
axes[2].set_title('Phase Spectrum (skimage)')
axes[2].axis('off')

# Show the plots
plt.show()
