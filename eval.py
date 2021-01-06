'''
Script to test a single model on the dataset.
'''

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from runtime_args import args
from model_factory.model_factory import ModelFactory
from load_dataset import LoadDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")


feature_level_train_data = LoadDataset(dataset_folder_path='./generated_dataset', level='feature_level', train=True, transform=transforms.ToTensor())
feature_level_test_data = LoadDataset(dataset_folder_path='./generated_dataset', level='feature_level', train=False, transform=transforms.ToTensor())

#if any of the arguments for the model initialization needs to be changed, it can be passed as a parameter here. The factory will override the default
#parameters for the models set in the model configuration files.
model, criterion, optimizer, lr_decayer = ModelFactory().factory_call(requested_model=args.model, mode=None,
                                                                      model_load_path=None, device=DEVICE, image_size=64)


feat_train_generator = DataLoader(feature_level_train_data, batch_size=10, shuffle=True, num_workers=3)

for i, data in enumerate(feat_train_generator):

    image, label = data['image'], data['label']
    print(image.cpu().numpy().shape)
    cv2.imshow(f'img-{str(label[0])}', image[0].cpu().numpy())
    cv2.waitKey(0)
