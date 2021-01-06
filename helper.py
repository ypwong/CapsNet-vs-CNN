'''
Helper functions.
'''
import os
import shutil
from tqdm import tqdm


def check_dir_exists(dir_path):
    '''
    Checks if a given directory exists.
    '''
    return os.path.exists(dir_path)

def del_dir(dir_path):
    '''
    Deletes the given directory.
    '''
    shutil.rmtree(dir_path)
    return None

def create_dir(dir_path):
    '''
    Creates a directory at the given path.
    '''
    os.makedirs(dir_path)
    return None

def run_cnn_model(generator, model, criterion, optimizer, lr_decayer, device, train=True):
    '''
    Use this func either to run one epoch of training or testing for the cnn model with the given data.
    '''


    epoch_loss = 0
    epoch_accuracy = 0
    i = 0
    for i, sample in tqdm(enumerate(generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        net_output = model(batch_x)

        if train:
            loss = model.optimize_model(predicted=net_output, target=batch_y, loss_func=criterion, optim_func=optimizer, decay_func=lr_decayer)
        else:
            loss = model.calculate_loss(predicted=net_output, target=batch_y, loss_func=criterion)

        epoch_loss += loss
        epoch_accuracy += model.calculate_accuracy(net_output, batch_y)


    return epoch_loss, epoch_accuracy, i


def run_capsnet_model(generator, model, criterion, optimizer, lr_decayer, device, train=True):
    '''
    Use this func either to run one epoch of training or testing for the cnn model with the given data.
    '''
    epoch_loss = 0
    epoch_accuracy = 0
    i = 0
    for i, sample in tqdm(enumerate(generator)):

        batch_x, batch_y = sample['image'].to(DEVICE), sample['label'].to(DEVICE)

        v_lengths, decoded_images = model(batch_x, batch_y)

        if train:
            loss = model.optimize_model(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images, loss_func=criterion,
                                            optim_func=optimizer, decay_func=lr_decayer, lr_decay_step=False)
        else:
            loss = model.calculate_loss(predicted=v_lengths, target=batch_y, ori_imgs=batch_x, decoded=decoded_images,
                                              loss_func=criterion)

        epoch_loss += loss
        epoch_accuracy += model.calculate_accuracy(v_lengths, batch_y)

    return epoch_loss, epoch_accuracy, i





