
from typing import Any
from PIL import Image
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

        self.ordered_pillows = []
        self.array = self.read(root = root, size = self.size, color_space = self.color_space, shuffle =True, should_save=True)


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __str__(self):

        info = f"# of images: {len(self.array)} in color space {self.color_space}, with shape {self.shape}. \n Preprocessed with so far using {self.preprocessor}."
        return info
    
    def read(self, root = None, size = (128, 128), color_space = 'RGB', shuffle =False, should_save = True):
        
        if root is None:
            raise ValueError("Please provide a valid root directory.")

        data = []

        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)
            if class_folder == 'numpy':
                continue

            if os.path.isdir(class_path):
                label = class_folder

                for file_name in tqdm(os.listdir(class_path), desc=str(label)):
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(class_path, file_name)

                        # Open and resize image
                        image = Image.open(image_path)
                        image = image.resize(size)

                        if color_space == 'RGB':
                            # Do nothing.
                            pass
                        else:
                            raise NotImplementedError()
                        
                        data.append([image, label, file_name])

        self.ordered_pillows = data
        array = np.array(data, dtype=object)
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
    
    def plot(self, random=True, size=10, color_space='RGB'):
        if random:
            indices = np.random.randint(len(self.ordered_pillows), size=size)
        else:
            indices = np.arange(min(size, len(self.ordered_pillows)))

        num_rows = size // 5  # Assuming you want 5 columns per row

        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_pillows[index]

            # Convert image to specified color space
            if color_space != 'RGB':
                raise NotImplementedError

            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':

    dataset = binaryDataset()

    dataset.plot()