import numpy as np
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk

def pearson_correlation(x, y):

    meanx = np.mean(x)
    meany = np.mean(y)

    corr = np.sum( (x-meanx) * (y-meany))
    norm = np.sqrt( np.sum((x-meanx) ** 2) * np.sum((y-meany) ** 2) )

    return corr / norm

def color_statistics(image, color_space = 'HSV'):

    """ 
    Outputs first order statistics of an image. 
    
    Inputs:
            image: ArrayLike (n, m, ch). Note: number of channels will be extracted from the last dimension. 
            
    Returns: A dictionary storing mean, standard variance, histogram per channels, skewness and kurtosis along with central moments.
    
    """
    channels = cv.split(image)

    fos = {}

    if color_space == 'RGB':
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        pass
    elif color_space == 'HSV':
        tmp = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    elif color_space == 'GRAY':
        pass
    elif color_space == 'LAB':
        tmp = cv.cvtColor(image, cv.COLOR_LAB2RGB)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    else:
        raise NotImplementedError()

    fos['contrast'] = np.std(gray)

    for i, ch in enumerate(channels):
        # Calculte histogram per channel.
        color = color_space[i]

        fos[f'{color}_mean'] = np.mean(ch)
        fos[f'{color}_std2'] = np.std(ch)
        
        moments = cv.moments(ch)

        mu20, mu11, mu02, mu30, mu21, mu12, mu03 = moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'], moments['mu21'], moments['mu12'], moments['mu03']

        fos[f'{color}_skewness'] = (mu30 * mu03 - 3 * mu12 * mu21) / (mu02**1.5 * mu20**2.5)
        fos[f'{color}_kurtosis'] = (mu30 * mu03 - 3 * mu12 * mu21) / (mu02**2 * mu20**2)

        fos[f'{color}_mu20'] = mu20
        fos[f'{color}_mu11'] = mu11
        fos[f'{color}_mu02'] = mu02
        fos[f'{color}_mu30'] = mu30
        fos[f'{color}_mu21'] = mu21
        fos[f'{color}_mu12'] = mu12
        fos[f'{color}_mu03'] = mu03

        fos[f'{color}_entropy'] = np.mean(entropy(ch, disk(5)) / ch.max())

    fos[f'corr_{color_space[0]}_{color_space[1]}'] = pearson_correlation(channels[0], channels[1])

    fos[f'corr_{color_space[0]}_{color_space[2]}'] = pearson_correlation(channels[0], channels[2])
    
    fos[f'corr_{color_space[1]}_{color_space[2]}'] = pearson_correlation(channels[1], channels[2])

    return fos

def color_hist_bins(image, n_bins = 256, color_space = 'HSV'):

    channels = cv.split(image)

    bins = {}

    for i, _ in enumerate(channels):
        # Calculte histogram per channel.
        hist = cv.calcHist(image, channels=[i], mask=None, histSize=[n_bins], ranges=[0, 256])

        color = color_space[i]
        for bin, height in enumerate(hist):
            bins[f'{color}_{bin}'] = height[0]
    
    return bins
    
# Current number of features: 
# RGB, HSV, LAB --> (1+ 3 + (12 + 256)*3)*3 = 2424. 
# Sadly, already close to the number of samples! (i.e., violation of the thumb rule of overfitting.)

# We should somehow decrease the number of bins, but one must do in a clever way. 
# Ideas: Superpixel-clustered density histogram, global thresholding, background subtraction, etc. 

# For now, we can just automatically decrease the sensitivity of histogram via n_bins argument. For example, 
# (1+ 3 + (12 + 50)*3)*3 = 570. Relatively better number of results. 

# I = cv.imread('../data/val/nev07726.jpg')
# color_hist_bins(I , n_bins=50)

