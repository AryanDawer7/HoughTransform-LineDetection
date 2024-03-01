import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    
    # Non-maximal suppression using dilation
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(H.astype('uint8'), kernel)
    
    # Match the local maxima with the dilated image
    local_maxima = (H == dilated)
    
    # Threshold to ensure we only consider strong local maxima
    threshold = np.sort(H[local_maxima].flatten())[-nLines]
    local_maxima[H < threshold] = False
    
    # Find indices of the local maxima
    indices = np.argwhere(local_maxima)

    # Sort indices by the accumulator values
    sorted_indices = indices[np.argsort(H[indices[:,0], indices[:,1]])[::-1]]
    
    # Select the top nLines strongest peaks
    strongest_peaks = sorted_indices[:nLines]
    
    # Extract rho and theta values for these peaks
    rhos = strongest_peaks[:,0]
    thetas = strongest_peaks[:,1]
    
    return rhos, thetas
