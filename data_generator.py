'''
Before training any model at all, do generate the necessary dataset.
'''

from tqdm import tqdm
import cv2
import helper
from dataset_factory.dataset_factory import DatasetFactory

def main(dataset_name, train_num_per_class, test_num_per_class, write_folder, overwrite_test=True, **kwargs):
    '''
    Generates the desired dataset both at feature-level and object-level. For testing mode, if there's already a dataset generated and if overwrite_test
    is set to True, then no test dataset will be generated. For experimentation purposes, there might be times when we want to generate diff amounts of
    training data while keeping the same testing data. This variable is to ensure that the test dataset won't be overwritten everytime a training set
    is intended to be generated. You can pass in parameters (hence the **kwargs) to overwrite the configuration of the generated data. However,
    check the available settings that are allowed to be modified in directly with the data generator functions.
    '''


    dataset_obj = DatasetFactory().factory_call(requested_dataset=dataset_name, **kwargs)
    feature_classes = dataset_obj.get_feature_level_data()
    object_classes = dataset_obj.get_object_level_data()

    print('--- Writing data to disk ---')
    for level in ['feature_level', 'object_level']:

        for mode in ['train', 'test']:

            class_ = feature_classes if level == 'feature_level' else object_classes
            write_path_mode = f"{write_folder.rstrip('/')}/{level}/{mode}/"

            if mode == 'train' and helper.check_dir_exists(write_path_mode):
                helper.del_dir(write_path_mode)

            #check if the testing data is to be generated.
            if mode == 'test' and helper.check_dir_exists(write_path_mode):
                if not overwrite_test:
                    continue
                helper.del_dir(write_path_mode)

            for class_idx, _ in enumerate(class_):

                write_path = f'{write_path_mode}{str(class_idx)}/'

                if not helper.check_dir_exists(write_path):
                    helper.create_dir(write_path)

                if mode == 'train':
                    for idx in tqdm(range(train_num_per_class)):

                        if level == 'feature_level':
                            image = dataset_obj.gen_feature_level(feature_index=class_idx)
                        else:
                            image = dataset_obj.gen_object_level(object_index=class_idx)

                        img_path = f'{write_path}{str(idx)}.jpg'

                        cv2.imwrite(img_path, image)

                elif mode == 'test':
                    for idx in tqdm(range(test_num_per_class)):

                        if level == 'feature_level':
                            image = dataset_obj.gen_feature_level(feature_index=class_idx)
                        else:
                            image = dataset_obj.gen_object_level(object_index=class_idx)

                        img_path = f'{write_path}{str(idx)}.jpg'

                        cv2.imwrite(img_path, image)

    print('--- Finish writing data to disk ---')


main(dataset_name='arrow_vs_nonarrow', train_num_per_class=500, test_num_per_class=1000, overwrite_test=True, write_folder='./generated_dataset')


