import numpy as np

def myImageFilter(img0, h):
    # Getting kernel dimensions
    kernel_height, kernel_width = h.shape

    # Pad the image relative to kernal dimensions
    padding_height, padding_width = kernel_height//2, kernel_width//2
    padded = np.pad(img0, ((padding_height, padding_height), (padding_width, padding_width)), 'edge')

    # Create blank image of the same size
    img1 = np.zeros_like(img0)

    # Looping via image pixels to apply the filter
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            # Increment through parts of the padded image
            region = padded[i:i+kernel_height, j:j+kernel_width]
            # Convolve with filter and store in result
            img1[i, j] = np.sum(np.multiply(region,h))
    
    return img1

########### UNIT TESTING (Uncomment to run)

# import cv2 

# ## Image processing
# img = cv2.imread('../data/img07.jpg')

# if (img.ndim == 3):
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = np.float32(img) / 255

# ## Kernels
# sharpen = np.array([[0, -1, 0],
#                     [-1, 5, -1],
#                     [0, -1, 0]])
# gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
#                                   [2., 4., 2.],
#                                   [1., 2., 1.]])

# # Function Testing

# img_gaussian = myImageFilter(img, gaussian)
# img_sharpen = myImageFilter(img, sharpen)

# cv2.imwrite('../results/img07_gaussian.png', 255 * img_gaussian)
# cv2.imwrite('../results/img07_sharpen.png', 255 * img_sharpen)