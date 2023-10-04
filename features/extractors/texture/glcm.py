from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import numpy as np

def calculate_glcms( image, distances = [8,16,32], angles = [0,np.pi/4,np.pi/2,3*np.pi/4], properties = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']):
    
    fos = {}
    angle_label = [0,45,90,135]
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    glcms = []
    
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(gray, [distance], [angle], levels=256, symmetric=True, normed=True)
            glcms.append(glcm)
    
    for i, glcm in enumerate(glcms):
        distance = distances[i // len(angles)]
        angle = angle_label[i % len(angles)]
        for prop in properties:
            fos[f'{prop}_{angle}_{distance}'] = graycoprops(glcm,prop).flat[0]
            
    return fos
    