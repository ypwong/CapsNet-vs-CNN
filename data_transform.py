'''
Transformation for the images.
'''

import numpy as np
import cv2
import torch

class Grayscale:
    '''
    Converts the RGB image to a grayscale image.
    '''

    def __call__(self, sample):
        '''
        Grayscale conversion.
        '''
        ori_image = sample['image']
        if ori_image.shape[2] > 1 :
            sample['image'] = np.expand_dims(cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY), axis=2)

        return sample

class ToTensor:
    '''
    Converts a numpy image into a tensor.
    '''

    def __call__(self, sample):
        '''
        Tensor conversion.
        '''

        image = sample['image'].transpose((2, 0, 1)) #pytorch requires the channel to be in the 1st dimension of the tensor.
        label = sample['label']

        return {'image': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                'label': torch.from_numpy(np.asarray(label, dtype='int32')).type(torch.LongTensor)}




