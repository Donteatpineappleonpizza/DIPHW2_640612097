import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def apply_butterworth_filter(image, cutoff_frequency, order=2):
    # Fourier Transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create a Butterworth filter mask
    X, Y = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - center_row)**2 + (Y - center_col)**2)
    butterworth_mask = 1 / (1 + (distance / cutoff_frequency)**(2*order))
    
    # Apply the Butterworth mask to the shifted Fourier spectrum
    filtered = f_transform_shifted * butterworth_mask
    
    # Inverse Fourier Transform
    f_ishift = ifftshift(filtered)
    img_back = ifft2(f_ishift)
    img_filtered = np.abs(img_back)
    
    return img_filtered


def calculate_rms_error(original, filtered):
    return np.sqrt(np.mean((original - filtered) ** 2))

def calculate_rms_error(original, filtered):
    return np.sqrt(np.mean((original - filtered) ** 2))

# Load the images
chess_noise = imread('Chess_noise.pgm')
lenna_noise = imread('Lenna_noise.pgm')
chess = imread('Chess.pgm')
lenna = imread('Lenna.pgm')

cutoff_frequencies = [10, 30, 50, 70]

# Process and calculate RMS for each cutoff frequency
for cutoff in cutoff_frequencies:
    chess_filtered = apply_butterworth_filter(chess_noise, cutoff)
    lenna_filtered = apply_butterworth_filter(lenna_noise, cutoff)
    
    chess_rms = calculate_rms_error(chess, chess_filtered)
    lenna_rms = calculate_rms_error(lenna, lenna_filtered)
    
    print(f'Chess RMS Error for cutoff {cutoff}: {chess_rms}')
    print(f'Lenna RMS Error for cutoff {cutoff}: {lenna_rms}')
    
# Adjustments for better inline display
plt.rcParams['figure.figsize'] = [8, 8]  # or another size that fits your screen

# Process and calculate RMS for each cutoff frequency and visualize
for cutoff in cutoff_frequencies:
    chess_filtered = apply_butterworth_filter(chess_noise, cutoff)
    lenna_filtered = apply_butterworth_filter(lenna_noise, cutoff)
    
    chess_rms = calculate_rms_error(chess, chess_filtered)
    lenna_rms = calculate_rms_error(lenna, lenna_filtered)
    
    # Print RMS error
    print(f'Chess RMS Error for cutoff {cutoff}: {chess_rms:.4f}')
    print(f'Lenna RMS Error for cutoff {cutoff}: {lenna_rms:.4f}')
    
    # Visualization
    fig, axs = plt.subplots(2, 3)
    
    axs[0, 0].imshow(chess, cmap='gray')
    axs[0, 0].set_title('Original Chess')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(chess_noise, cmap='gray')
    axs[0, 1].set_title('Chess with Noise')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(chess_filtered, cmap='gray')
    axs[0, 2].set_title(f'Filtered Chess (Cutoff: {cutoff})')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(lenna, cmap='gray')
    axs[1, 0].set_title('Original Lenna')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(lenna_noise, cmap='gray')
    axs[1, 1].set_title('Lenna with Noise')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(lenna_filtered, cmap='gray')
    axs[1, 2].set_title(f'Filtered Lenna (Cutoff: {cutoff})')
    axs[1, 2].axis('off')
    
    plt.show()