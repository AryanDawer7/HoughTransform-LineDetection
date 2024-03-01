import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):

    # Dims of the image
    height, width = Im.shape
    
    # Max rho
    rhoMax = round(np.sqrt(height**2 + width**2))
    
    # Create rho and theta scales
    rhoScale = np.arange(0, rhoMax, rhoRes)
    thetaScale = np.arange(0, np.pi, thetaRes)
    
    # Initialize the Hough accumulator array to zero
    num_rhos = len(rhoScale)
    num_thetas = len(thetaScale)
    img_hough = np.zeros((num_rhos, num_thetas), dtype=int)
    
    # Find all the edge pixel indexes
    y_idxs, x_idxs = np.nonzero(Im)
    
    # Vote in the Hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(num_thetas):
            # Calculate rho
            rho = round((x * np.cos(thetaScale[j]) + y * np.sin(thetaScale[j])) / rhoRes) * rhoRes
            rho_index = np.nonzero(rhoScale == rho)[0]
            
            # If the rho is within valid range, cast a vote
            if rho >= 0 and len(rho_index) == 1:
                img_hough[rho_index[0], j] += 1
    
    return img_hough, rhoScale, thetaScale