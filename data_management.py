import pandas as pd
import skimage.io
import torch
import numpy as np


class data_management:

    def __init__(self, data_path, train_test_split = 0.8, shuffle = True, dimension = 224, transformations = False, batch_size = 32, device = 'mps'):
        self.data_path = data_path
        self.train_test_split = train_test_split
        self.shuffle = shuffle
        self.dimension = dimension
        self.transformations = transformations
        self.batch_size = batch_size
        self.labels = pd.read_csv("../image_labels.csv")


    def load_data(self, image_tensor, labels_tensor, train_test_split, shuffle, transformations, batch_size):
        if(self.transformations):
            train_data_tensor = self.transform_images(image_tensor)
        


    def transform_images(self, image_tensor):
        return image_tensor

    def process_images(self):
        image_list = []
        image_labels = []

        for image_path in self.data_path:
            image = self.preprocess_image(image_path, dimension=self.dimension)
            image_label = self.get_image_labels(image_path)
            image_list.append(image)
            image_labels.append(image_label)

        # Convert lists to tensors here if necessary
        image_tensor = torch.stack(image_list)  # Assuming you convert the list to a tensor
        labels_tensor = torch.tensor(image_labels, dtype=torch.long)

        # Assuming `load_data` is a static method in `data_management` that expects tensors
        train_data_tensor, train_labels, test_data_tensor, test_labels = data_management.load_data(
            image_tensor, labels_tensor, self.train_test_split, self.shuffle, transformations=self.transformations, batch_size=self.batch_size
        )
        return train_data_tensor, train_labels, test_data_tensor, test_labels
        

    def get_image_labels(self, data_path):
        for label in self.labels[""]:
            if data_path in label:
                return label
        

    def preprocess_image(image_path, dimension = 224):
        image = skimage.io.imread(image_path)
        image = skimage.transform.resize(image, (dimension, dimension))
        return image