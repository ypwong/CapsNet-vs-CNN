'''
Handles the loading of the dataset.
'''
import glob
import cv2
import torch
from torch.utils.data import Dataset
import helper

class LoadDataset(Dataset):
    '''
    Loads the dataset from the given path.
    '''

    def __init__(self, dataset_folder_path, level, train=True, transform=None):
        '''
        Initialize parameters and perform checks on the data.
        '''

        assert not dataset_folder_path is None, "Path to the data folder must be provided."
        assert level == 'feature_level' or level == 'object_level', 'Please provide an appropriate level name. Either "object_level" or "feature_level".'

        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.level = level
        self.train = train

        self.image_path_label = self.read_folder()



    def read_folder(self):
        '''
        Reads the folder for the images and their labels.
        '''

        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/{self.level}/train/"
        else:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/{self.level}/test/"

        assert helper.check_dir_exists(folder_path), "The specified directory does not exist! Aborting ..."

        for x in glob.glob(folder_path + '**', recursive=True):

            if not x.endswith('jpg'):
                continue

            class_idx = x.split('/')[-2]

            image_path_label.append((x, int(class_idx)))

        return image_path_label


    def __len__(self):
        '''
        Returns the num of total files.
        '''

        return len(self.image_path_label)

    def __getitem__(self, idx):
        '''
        Returns a single image and its corresponding label.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.image_path_label[idx]

        image = cv2.imread(image)

        sample = {
            'image': image,
            'label':label
        }

        if self.transform :
            sample['image'] = self.transform(sample['image'])

        return sample
