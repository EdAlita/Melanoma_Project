import numpy as np
from tqdm import tqdm
import pandas as pd 
from extractors.color import color

def extract_features(stack, descriptors=None, save=True, feature_dir="features/all/"):

    """Extract features from input images using specified feature extractors."""

    stack = np.array(stack, dtype=object)
    # Get images.
    images = stack[:, 0]
    # Get targets.
    labels = stack[:, 1]
    # Get filenames
    fnames = stack[:, 2]

    # Initialize a data frame with column titles to save feature points later on. 
    initial_img = images[0]

    # Initial run.
    cols = ['fname']

    for descriptor in descriptors:

        feature_dict = descriptor(initial_img)
        
        for col in list(feature_dict.keys()):
            cols.append(col)

    cols.append('label')
    
    dataframe = pd.DataFrame(columns=cols)

    print(dataframe)

    # Loop for feature extraction.
    for index, image in tqdm(enumerate(images)):

        row = {}        
        for descriptor in descriptors:
            tmp = descriptor(image)

            # Merge dictionary with row.
            row.update(tmp)  # Merge 'tmp' dictionary into 'row'

        row['fname'] = fnames[index]
        row['label'] = labels[index]

        # Add row as a row to pandas data frame `dataframe`.
        dataframe = dataframe.append(row, ignore_index=True)


    if save:
        filename = feature_dir + 'features_hsv.csv'
        dataframe.to_csv(filename, index=False)

    return dataframe


if __name__ == '__main__':

    import os, sys
    # Get the parent directory path
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
    # Add the parent directory to the Python path
    sys.path.append(parent_dir)
    from dataset import binaryDataset

    # Uncomment for test.
    # import cv2 as cv

    # image1, image2 = cv.imread('data/val/nevus/nev07726.jpg'), cv.imread('data/val/others/ack00521.jpg')

    # image1, image2 = cv.cvtColor(image1, cv.COLOR_BGR2RGB), cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    # label1, label2 = 'nevus', 'others'

    # fname1, fname2 = 'nev07726', 'ack00521'

    # stack =np.asarray( [[image1, label1, fname1], [image2, label2, fname2]] , dtype=object)

    # dataframe = extract_features(stack, descriptors=[color.color_statistics, color.color_hist_bins])

    dataset = binaryDataset(color_space='HSV')

    dataframe = extract_features(dataset.ordered_images, descriptors=[color.color_statistics, color.color_hist_bins])
