
from typing import Any
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

class binaryDataset():
    
    def __init__(self, 
                 root = 'data/val/',
                 img_size = (128, 128),  # Image size.
                 n_channels = 3, # Whether we use gray or RGB, or whether we couple it with another space.
                 preprocessor = None, # Whether and in which way we preprocess while calling the dataset.
                 color_space = 'RGB' # Which color space we are inspecting on. 
                 ):
        
        self.mode = 'binary' # That's fixed.
        self.size = img_size
        self.n_channels = n_channels
        self.preprocessor = preprocessor
        self.color_space = color_space
        
        shape = (img_size[0], img_size[1], n_channels)
        self.shape = shape

        self.ordered_images = []
        self.array = self.read(root = root, size = self.size, color_space = self.color_space, shuffle =True, should_save=True)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __str__(self):

        info = f"# of images: {len(self.array)} in color space {self.color_space}, with shape {self.shape}. \n Preprocessed with so far using {self.preprocessor}."
        return info
    
    def read(self, root = None, size = (128, 128), color_space = 'RGB', shuffle =True, should_save = True):
        
        if root is None:
            raise ValueError("Please provide a valid root directory.")

        data = []

        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)
            if class_folder == 'numpy':
                continue

            if os.path.isdir(class_path):
                label = class_folder

                for file_name in tqdm(set(os.listdir(class_path)), desc=str(label)):
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(class_path, file_name)

                        # Open and resize image
                        image = cv.imread(image_path)
                        image = cv.resize(image, size)

                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                        if color_space == 'RGB':
                            # Do nothing.
                            pass

                        elif color_space == 'HSV':
                            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

                        elif color_space == 'GRAY':
                            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

                        elif color_space == 'LAB':
                            image = cv.cvtColor(image, cv.COLOR_RGB2LAB)

                        else:
                            raise NotImplementedError()
                        
                        fname, jpg = file_name.split('.')
                        data.append([image, label, fname])

        self.ordered_images = data
        array = np.array(data, dtype=object)

        if shuffle:
            np.random.shuffle(array)


        return array

    def resize(self, should_save = False):
        """Resize images and return numpy arrays again. """

        return self.images
    
    def preprocess(self, should_save = False):
        """Preprocess images and return numpy arrays again. """

        return self.images
    
    def plot(self, random=True, size=5, color_space='None', ch=0):
        if random:
            indices = np.random.randint(len(self.ordered_images), size=size)
        else:
            indices = np.arange(min(size, len(self.ordered_images)))

        num_rows = size // 5  # Assuming you want 5 columns per row

        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')

        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            # Let's have a look at the H channel.
            # Split the image into its RGB channels
            ch1, ch2, ch3 = cv.split(image)

            # Calculate the histograms
            hist_ch1 = cv.calcHist([ch1], [0], None, [256], [0, 256])
            hist_ch2 = cv.calcHist([ch2], [0], None, [256], [0, 256])
            hist_ch3 = cv.calcHist([ch3], [0], None, [256], [0, 256])

            # Plot the histograms
            axes[i].plot(hist_ch1, color='pink')
            axes[i].plot(hist_ch2, color='orange')
            axes[i].plot(hist_ch3, color='brown')
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')
            
            
        fig, ax = plt.subplots(3, 5, figsize=(15, 3*3))

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            # Let's have a look at the H channel.
            # Split the image into its RGB channels
            ch1, ch2, ch3 = cv.split(image)
            
            ax[0,i].imshow(ch1)
            ax[0,i].set_title(f'Label: {label}\nFile Name: {file_name} ch1')
            ax[0,i].axis('off')
            ax[1,i].imshow(ch2)
            ax[1,i].set_title(f'Label: {label}\nFile Name: {file_name} ch2')
            ax[1,i].axis('off')
            ax[2,i].imshow(ch3)
            ax[2,i].set_title(f'Label: {label}\nFile Name: {file_name} ch3')
            ax[2,i].axis('off')
            

        plt.tight_layout()
        plt.show()


class mcDataset():
    
    def __init__(self, 
                 root = 'data/val/',
                 img_size = (128, 128),  # Image size.
                 n_channels = 3, # Whether we use gray or RGB, or whether we couple it with another space.
                 preprocessor = None, # Whether and in which way we preprocess while calling the dataset.
                 color_space = 'RGB' # Which color space we are inspecting on. 
                 ):
        
        self.mode = 'mc' # That's fixed.
        self.size = img_size
        self.n_channels = n_channels
        self.preprocessor = preprocessor
        self.color_space = color_space
        
        shape = (img_size[0], img_size[1], n_channels)
        self.shape = shape

        self.ordered_images = []
        self.array = self.read(root = root, size = self.size, color_space = self.color_space, shuffle =True, should_save=True)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __str__(self):

        info = f"# of images: {len(self.array)} in color space {self.color_space}, with shape {self.shape}. \n Preprocessed with so far using {self.preprocessor}."
        return info
    
    def read(self, root = None, size = (128, 128), color_space = 'RGB', shuffle =True, should_save = True):
        
        if root is None:
            raise ValueError("Please provide a valid root directory.")

        data = []

        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)
            if class_folder == 'numpy':
                continue
            
            if os.path.isdir(class_path):

                if class_folder == 'nevus':
                    label = class_folder
                else:
                    label = 'others'

                for file_name in tqdm(set(os.listdir(class_path)), desc=str(label)):
                    label_others = np.array(['ack', 'bcc', 'bkl', 'def', 'mel', 'scc', 'vac'])

                    for _ in label_others:
                        if _ in file_name:
                            label = _
                            break
                    
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(class_path, file_name)

                        # Open and resize image
                        image = cv.imread(image_path)
                        image = cv.resize(image, size)

                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                        if color_space == 'RGB':
                            # Do nothing.
                            pass

                        elif color_space == 'HSV':
                            image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

                        elif color_space == 'GRAY':
                            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

                        elif color_space == 'LAB':
                            image = cv.cvtColor(image, cv.COLOR_RGB2LAB)

                        else:
                            raise NotImplementedError()
                        
                        fname, jpg = file_name.split('.')
                        data.append([image, label, fname])

        self.ordered_images = data
        array = np.array(data, dtype=object)

        if shuffle:
            np.random.shuffle(array)

        if should_save:

            save_path = f"{root}/numpy/" 
            if save_path is None:
                raise ValueError("Please provide a valid save path.")

            np.save(save_path, np.array(data, dtype=object))

        return array

    def resize(self, should_save = False):
        """Resize images and return numpy arrays again. """

        return self.images
    
    def preprocess(self, should_save = False):
        """Preprocess images and return numpy arrays again. """

        return self.images
    
    def plot(self, random=True, size=5, color_space='None', ch=0):
        if random:
            indices = np.random.randint(len(self.ordered_images), size=size)
        else:
            indices = np.arange(min(size, len(self.ordered_images)))

        num_rows = size // 5  # Assuming you want 5 columns per row

        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')

        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            # Let's have a look at the H channel.
            # Split the image into its RGB channels
            ch1, ch2, ch3 = cv.split(image)

            # Calculate the histograms
            hist_ch1 = cv.calcHist([ch1], [0], None, [256], [0, 256])
            hist_ch2 = cv.calcHist([ch2], [0], None, [256], [0, 256])
            hist_ch3 = cv.calcHist([ch3], [0], None, [256], [0, 256])

            # Plot the histograms
            axes[i].plot(hist_ch1, color='pink')
            axes[i].plot(hist_ch2, color='orange')
            axes[i].plot(hist_ch3, color='brown')
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')
            
            
        fig, ax = plt.subplots(3, 5, figsize=(15, 3*3))

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Convert image to specified color space
            if color_space != 'None':
                raise NotImplementedError

            # Let's have a look at the H channel.
            # Split the image into its RGB channels
            ch1, ch2, ch3 = cv.split(image)
            
            ax[0,i].imshow(ch1)
            ax[0,i].set_title(f'Label: {label}\nFile Name: {file_name} ch1')
            ax[0,i].axis('off')
            ax[1,i].imshow(ch2)
            ax[1,i].set_title(f'Label: {label}\nFile Name: {file_name} ch2')
            ax[1,i].axis('off')
            ax[2,i].imshow(ch3)
            ax[2,i].set_title(f'Label: {label}\nFile Name: {file_name} ch3')
            ax[2,i].axis('off')
            

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    dataset = mcDataset(color_space='RGB')

    dataset.plot()