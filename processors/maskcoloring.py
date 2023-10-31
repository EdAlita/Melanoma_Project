import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os,sys
# # get the parent directory path
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# # add the parent directory to the Python path
# sys.path.append(parent_dir)

import tools.mfr as f
import tools.morph as m
from patch_based_inpainting.inpaint import *

def preprocess(path, save = True, 
               out_dir = 'data/preprocessed/train/', 
               mask_dir = 'data/hair/train/', 
               mean_dir = 'data/meancolored/train/',
               show = True, ralg = 'telea'):
    
    lname, fname = path.split('\\')[-2:]
    
    mean_colored_path = mean_dir + '/' + lname + '/' + fname

    if os.path.exists(mean_colored_path):
        return 
    
    # print("NAME", lname, fname)
    # Step 0.
    im = np.array(cv2.imread(path, cv2.IMREAD_COLOR))

    im0 =  cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Step 1.
    L = 6     # the length of the neighborhood along the y-axis to smooth noise
    sigma = 1.5 # (?)# scale
    w = 31 # (?)    # kernel size
    c = 4 # the gain of threshold
    n = 130 # number of pixels to eliminate
    rot = 12 # number of rotations

    params = [L, sigma, w, c, n, rot]

    out, H = f.pipeline(path, params = params, filter_out = False, save=False)

    # Step 2.
    mask, maskf = m.cls(out, r = 5, filter_out= 100, eps=0.01)

    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maskf.astype(np.uint8), connectivity=4)

    # def calc_rdst(x, y, center):
    #     # calculate the distance metric (rdst) based on (x, y) and center
    #     return np.sqrt((x - center[0])**2 + (y - center[1])**2)


    # # loop through each labeled region
    # hair_labs = []
    # for label in range(1, num_labels):  # skip the background label (0)
    #     # create a mask for the current
    #     label_mask = (labels == label).astype(np.uint8)

    #     # extract the statistics
    #     stats_label = stats[label]

    #     # get the bbox coordinates (x, y, width, height)
    #     x, y, w, h = stats_label[0], stats_label[1], stats_label[2], stats_label[3]

    #     # crop the region from the original image
    #     cropped_region = label_mask[y:y+h, x:x+w]

    #     center = cropped_region.shape[0] / 2, cropped_region.shape[1] / 2

    #     y_coords, x_coords = np.where(cropped_region == 1)

    #     rdst = [calc_rdst(x, y, center = center) for x, y in  zip(x_coords, y_coords)]
        
    #     hcirc = np.mean(rdst) / np.std(rdst) if np.std(rdst) !=0 else 1 

    #     # plt.imshow(cropped_region)

    #     meanc =  hcirc * np.mean(cropped_region)

    #     # print(hcirc, meanc)

    #     if meanc <= 0.4:
    #         # assume it is a hair.
    #         hair_labs.append(label)

    # # Step 3 Superimpose & Restoration.
    # hair_mask = np.isin(labels, hair_labs).astype(np.uint8) * 255

  
    mean_per_chs = np.mean(im0, axis=(0,1))

    res = im0.copy()
    res[maskf > 0] = mean_per_chs


    # if ralg == 'telea':
    #     dst = cv2.inpaint(im0, mask.astype(np.uint8), 10, cv2.INPAINT_TELEA)


    if show:    
        fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        axs = axs.ravel()
        plt.tight_layout()

        images = [im0, im, H, out, mask, maskf, out-mask, mask-maskf, res]
        titles = ['Color', 'Gray', 'Response to MF-FDOG', 'Thresholded', 'Closed', 'Filtered-Out', 'Difference of Filter-Out', 
                'Difference of Filtered', 'Superimposed']
        
        for i in range(len(axs)):
            if titles[i] == 'CC-Labels':
                axs[i].imshow(images[i], cmap = 'rainbow')
            else:
                axs[i].imshow(images[i])
            axs[i].set_title(titles[i])
            axs[i].axis("off")

        plt.show()

    
    if save:

        # out_path = out_dir + '/' + lname + '/' + fname

        # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(out_path, dst)

        mask_path = mask_dir + '/' + lname + '/' + fname
        cv2.imwrite(mask_path, maskf)

        mean_colored_path = mean_dir + '/' + lname + '/' + fname
        cv2.imwrite(mean_colored_path, res)
        # print(mean_colored_path)

    return res

if __name__ == '__main__':

    from tqdm import tqdm 

    # define the directory path
    directory_path = "./data/train"
    print(os.listdir(directory_path))
    # initialize an empty list to store the file paths
    file_paths = []

    # iterate through the files in the directory
    for folder in os.listdir(directory_path):
        for filename in os.listdir(os.path.join(directory_path, folder)):
            # check if the entry is a file (not a directory)
            if os.path.isfile(os.path.join(directory_path, folder, filename)):
                # construct the full path and add it to the list
                file_path = os.path.join(directory_path, folder, filename)
                # print(file_path)
                file_paths.append(file_path)

    # run loop 
    for path in tqdm(file_paths):
        try:
            preprocess(path, show=False, save=True)
        except Exception as e:
            print(path)
            pass