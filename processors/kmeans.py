import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import preprocessing as p

def kmeans_segmentation(image, num_clusters=3):   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # Flatten the image
    pixels = image.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    
    # Reshape the labels to match the original image shape
    cluster_labels = labels.reshape(image.shape[:2])
    
    # Find the cluster with the largest number of pixels
    unique, counts = np.unique(cluster_labels, return_counts=True)
    largest_cluster_label = unique[np.argmax(counts)]
    
    # Create a mask for the largest cluster
    mask = (cluster_labels == largest_cluster_label).astype(np.uint8)
    
    return mask

def apply_mask_to_image(image, mask):
  
    # Apply the mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb

def superpixel_segmentation(image, num_segments=100):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply SLIC superpixel segmentation
    segments = slic(image, n_segments=num_segments, compactness=10)
    
    # Convert segments to RGB format
    segments_rgb = label2rgb(segments, image, kind='avg')
    
    # Convert segments to grayscale for mask creation
    mask = cv2.cvtColor(segments_rgb.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    return mask

def extract_largest_cluster(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask of the same size as the original mask
    largest_cluster_mask = np.zeros_like(mask)
    
    # Draw the largest contour on the empty mask
    cv2.drawContours(largest_cluster_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return largest_cluster_mask


# Example usage
image_path = "./data/val/nevus/nev07762.jpg"
num_clusters = 5  # You can adjust this based on your needs

# Load the image
image = cv2.imread(image_path)

## Preprocess: DullRazor
#Gray scale
grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
#Black hat filter
kernel = cv2.getStructuringElement(1,(9,9)) 
blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
#Gaussian filter
bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
#Binary thresholding (MASK)
ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
#Replace pixels of the mask
dst = cv2.inpaint(image,mask,6,cv2.INPAINT_TELEA)   

## Preprocess: Pipeline
dst1 = p.preprocess(image_path, ralg = 'telea', show=False, save=False, out_dir = 'data/preprocessed/val/')


# Apply K-means segmentation and get the largest cluster mask
mask0 = kmeans_segmentation(image, num_clusters)
# Apply the mask to the original image
result0 = apply_mask_to_image(image, mask0)

# Apply K-means segmentation and get the largest cluster mask
mask1 = kmeans_segmentation(dst1, num_clusters)
# Apply the mask to the original image
result1 = apply_mask_to_image(dst1, mask1)

# Apply K-means segmentation and get the largest cluster mask
mask = kmeans_segmentation(dst, num_clusters)
# Apply the mask to the original image
result = apply_mask_to_image(dst, mask)


# Example usage
num_segments = 100  # You can adjust this based on your needs

# Apply K-means segmentation and get the largest cluster mask
mask0s = extract_largest_cluster(superpixel_segmentation(image, num_segments))
# Apply the mask to the original image
result0s = apply_mask_to_image(image, mask0s)

# Apply K-means segmentation and get the largest cluster mask
mask1s = extract_largest_cluster(superpixel_segmentation(dst1, num_segments))
# Apply the mask to the original image
result1s = apply_mask_to_image(dst1, mask1s)

# Apply K-means segmentation and get the largest cluster mask
mask_s = extract_largest_cluster(superpixel_segmentation(dst, num_segments))
# Apply the mask to the original image
result_s = apply_mask_to_image(dst, mask_s)


# Display the result
plt.subplot(2, 3, 1)
plt.imshow(result)
plt.subplot(2, 3, 2)
plt.imshow(result0)
plt.subplot(2, 3, 3)
plt.imshow(result1)

# Display the result
plt.subplot(2, 3, 4)
plt.imshow(mask_s)
plt.subplot(2, 3, 5)
plt.imshow(mask0s)
plt.subplot(2, 3, 6)
plt.imshow(mask1s)

plt.show()
# from tqdm import tqdm 

# # define the directory path
# directory_path = "./data/val"
# print(os.listdir(directory_path))
# # initialize an empty list to store the file paths
# file_paths = []

# # iterate through the files in the directory
# for folder in os.listdir(directory_path):
#     for filename in os.listdir(os.path.join(directory_path, folder)):
#         # check if the entry is a file (not a directory)
#         if os.path.isfile(os.path.join(directory_path, folder, filename)):
#             # construct the full path and add it to the list
#             file_path = os.path.join(directory_path, folder, filename)
#             # print(file_path)
#             file_paths.append(file_path)

# # run loop 
# for path in tqdm(file_paths):
#     preprocess(path, ralg = 'telea', show=False, save=True, out_dir = 'data/preprocessed/val/')