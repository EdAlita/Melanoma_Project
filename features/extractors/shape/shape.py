import numpy as np
import cv2 as cv

def shape_measurements(image,color_space='HSV'):
    fos = {}
    
    Area,Perimeter = 0,0
    
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
    
    ret, thresh = cv.threshold(gray,127,255,0)
    countours, hierarchy = cv.findContours(thresh, 1, 2)
    
    for cnt in countours:    
        Area =+ cv.contourArea(cnt)
        Perimeter =+ cv.arcLength(cnt,True)
        
    fos['Dispersity'] = Perimeter**2/Area
    fos['Saturation'] = Area/Perimeter
    fos['Roundness']  = 4*np.pi*Area/Perimeter**2

    return fos