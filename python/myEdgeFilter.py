import numpy as np
from scipy import signal    # For signal.gaussian function
from math import ceil

# import cv2 # For testing

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    # Creating Gaussian kernal
    hsize = 2 * ceil(3 * sigma) + 1
    gaussian_1d = signal.windows.gaussian(hsize, sigma)
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)
    gaussian_kernel /= np.sum(gaussian_kernel)
    
    # Convolving with gaussian kernel
    smoothed_img = myImageFilter(img0, gaussian_kernel)
    
    # cv2.imshow("Gaussian Blur", smoothed_img) # Testing

    # Definining sobel filters and convolving to find x and y gradients of img
    sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    sobel_y = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])

    imgx = myImageFilter(smoothed_img, sobel_x)
    imgy = myImageFilter(smoothed_img, sobel_y)

    # cv2.imshow("imgX", imgx) # Testing
    # cv2.imshow("imgY", imgy) # Testing

    # Calculating edge magnitude and orientation (angle in degrees)
    edge_magnitude = np.sqrt(imgx**2 + imgy**2)
    edge_orientation = np.rad2deg(np.arctan2(imgy, imgx)) % 180

    # Implementing the algorithm for Non-maximum suppression
    img1 = np.zeros_like(edge_magnitude) # create a result img
    for i in range(1, img1.shape[0] - 1):
        for j in range(1, img1.shape[1] - 1): # iterate via the pixels
            # Get the orientation at that point
            angle = edge_orientation[i,j] 
            
            # Map the gradient orientation to the closest 0, 45, 90, or 135 degrees
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [edge_magnitude[i, j-1], edge_magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [edge_magnitude[i-1, j+1], edge_magnitude[i+1, j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [edge_magnitude[i-1, j], edge_magnitude[i+1, j]]
            else:
                neighbors = [edge_magnitude[i-1, j-1], edge_magnitude[i+1, j+1]]

            # Finally suppress non-maximum pixels
            if edge_magnitude[i, j] >= max(neighbors):
                img1[i, j] = edge_magnitude[i, j]
    
    # cv2.imshow("Edges", img1) # Testing
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img1


########## UNIT TESTING (Uncomment to run)

# import cv2 

# ## Image processing
# img = cv2.imread('../data/img07.jpg')

# if (img.ndim == 3):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = np.float32(img) / 255

# # Function Testing

# cv2.imwrite('../results/img07_edges.png', 255 * myEdgeFilter(img, 2))