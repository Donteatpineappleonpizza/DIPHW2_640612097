from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def pad_to_next_power_of_2(image):
    """
    Pad the image to the next power of 2 in both dimensions.
    """
    m, n = image.shape
    M, N = 2**np.ceil(np.log2([m,n])).astype(int)
    padded = np.zeros((M,N), dtype=image.dtype)
    
    startx, starty = (M - m) // 2, (N - n) // 2  # Calculate starting points
    
    padded[startx:startx+m, starty:starty+n] = image  # Place image in the center
    return padded

# Load the image
image_path = 'Cross.pgm'
image = imread(image_path)

# Pad the image
padded_image = pad_to_next_power_of_2(image)

# Display the original and padded images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original image
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('on')

# Padded image
ax[1].imshow(padded_image, cmap='gray')
ax[1].set_title('Padded Image')
ax[1].axis('on')

plt.show()

# Perform Fourier Transform on the padded image
fft_image = fft2(padded_image)
fft_shifted = fftshift(fft_image)  # Shift the zero frequency component to the center

# Calculate amplitude and phase spectra
amplitude_spectrum = np.abs(fft_shifted)
phase_spectrum = np.angle(fft_shifted)

# Display the amplitude and phase spectra
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Amplitude spectrum
ax[0].imshow(np.log1p(amplitude_spectrum), cmap='gray')
ax[0].set_title('Amplitude Spectrum')
ax[0].axis('off')

# Phase spectrum
ax[1].imshow(phase_spectrum, cmap='gray')
ax[1].set_title('Phase Spectrum')
ax[1].axis('off')

plt.show()


from numpy.fft import ifft2, ifftshift

def shift_image_via_fft(fft_data, dx, dy):
    """
    Shift the image by dx and dy using phase manipulation in the frequency domain.
    
    Parameters:
    - fft_data: The FFT of the image.
    - dx: The shift in the x direction.
    - dy: The shift in the y direction.
    
    Returns:
    - The inverse FFT of the shifted image.
    """
    (rows, cols) = fft_data.shape
    M, N = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Create the phase shift array
    phase_shift = np.exp(-2j * np.pi * ((dx * M / cols) + (dy * N / rows)))
    
    # Apply the phase shift
    shifted_fft_data = fft_data * phase_shift
    
    # Perform inverse FFT
    shifted_image = ifft2(shifted_fft_data)
    
    return shifted_image

# Shift the image by (20,30) in the x and y directions, respectively
shifted_image = shift_image_via_fft(fft_shifted, 20, 30)

# Plot the shifted image
plt.figure(figsize=(6, 6))
plt.imshow(np.abs(shifted_image), cmap='gray')
plt.title('Shifted Image')
plt.axis('on')
plt.show()

