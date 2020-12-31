'''
Handles the loading of the required dataset and pass it to the machine learning model.
'''

import sys
from tqdm import tqdm
import os
import shutil
import glob
import cv2
import torch
from torch.utils.data import Dataset
from dataset_gen_files.dataset_factory import DatasetFactory


class LoadDataset(Dataset):
    '''
    Loads the dataset from the given path or generate the dataset, write it into disk and returns the data.
    '''
    def __init__(self, dataset_name, dataset_level, num_data_per_class=None, load_from_disk=False, data_path=None, train=True, transform=None, **kwargs):

        assert not data_path is None, "A path must be provided for the data to be written to. If you wish to load the dataset from a folder, provide the \
            path to the folder and set `load_from_disk` to True."

        dataset_name = dataset_name

        self.train = train
        self.data_path = data_path
        self.load_from_disk = load_from_disk
        self.dataset_level = dataset_level
        self.num_data_per_class = num_data_per_class
        self.transform = transform

        if self.train:
            self.data_path = self.data_path + '/train'
        else:
            self.data_path = self.data_path + '/test'

        self.data_path = self.data_path.replace('//', '/')#if the user supplied '/' at the end of the path, then there will be double slahes.

        if self.load_from_disk:

            self.dataset_object = DatasetFactory().factory_call(requested_dataset=dataset_name, **kwargs)
            if self.dataset_level == 'feature_level':
                self.classes = self.dataset_object.get_feature_level_data()
            elif self.dataset_level == 'object_level':
                self.classes = self.dataset_object.get_object_level_data()

            self.image_label_path = self.images_retriever()
            self.total_data = len(self.image_label_path)

        else:

            self.dataset_object = DatasetFactory().factory_call(requested_dataset=dataset_name, **kwargs)
            if self.dataset_level == 'feature_level':
                self.classes = self.dataset_object.get_feature_level_data()
            elif self.dataset_level == 'object_level':
                self.classes = self.dataset_object.get_object_level_data()

            self.prep_folder()
            self.image_label_path = self.write_images()



    def prep_folder(self):
        '''
        In order to conduct this experiment with different size of datasets, the folder has to be cleared everytime the dataset is generated.
        '''

        #creates the folder if it does not exist.
        if not os.path.exists(self.data_path) :
            os.makedirs(self.data_path)
            return None
        try:
            shutil.rmtree(self.data_path)
        except Exception as e:
            print("Contents in the specified folder cannot be deleted!")
            print(e)
            sys.exit()

        os.makedirs(self.data_path)
        return None


    def images_retriever(self):
        '''
        Returns a list of tuples with path of the images and their corresponding class index.
        '''


        image_path_label = []

        for x in glob.glob(self.data_path+"/**", recursive=True):

            if x[-3:] != 'jpg' :
                continue

            path_name = x[len(self.data_path):]

            #if path starts with '/', the first item is ''
            folder_name = path_name.split('/')[0] if path_name.split('/')[0] != '' else path_name.split('/')[1]

            class_name_bool = False #to check whether the folder name is valid or not.

            for class_alphabet, class_numeric in self.classes.items():

                if folder_name == class_alphabet or folder_name == str(class_numeric):
                    class_name_bool = True
                    image_path_label.append((x, class_numeric))

            if not class_name_bool:
                raise NameError("The name of the folders has to be the exact match of the class names either alphabetically or numerically!")

        return image_path_label


    def write_images(self):
        '''
        Writes the generated images on disk and returns the list of path and classes.
        '''
        image_path_label = []

        print("--- Writing data to disk ---")
        for i in tqdm(range(self.__len__())):

            #class_index is to ensure that data from all the classes in that particular level is generated and none is left out.
            #since idx will be from 0 to total number of images, dividing it by number of items in every class will provide the index of each class
            #for the given idx.
            class_index = i//self.num_data_per_class

            save_path = self.data_path + '/' + str(class_index) +'/'

            if not os.path.exists(save_path) : os.makedirs(save_path)

            if self.dataset_level == 'feature_level':
                image = self.dataset_object.gen_feature_level(feature_index=class_index)
            else:
                image = self.dataset_object.gen_object_level(object_index=class_index)

            img_path = save_path + str(i - (class_index*self.num_data_per_class))+'.jpg'
            image_path_label.append((img_path, class_index))
            #write the image in the given path where the name of the files will be the index value from 0 to self.num_data_per_class - 1
            cv2.imwrite(img_path, image)

        return image_path_label


    def __len__(self):
        '''
        Returns the total number of images to be generated/read.
        '''
        if self.load_from_disk:
            return self.total_data

        return self.num_data_per_class*len(self.classes)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.image_label_path[idx]
        image = cv2.imread(image)

        sample = {
                'image':image,
                'label':label
            }


        if self.transform :
            sample = self.transform(sample)

        return sample

