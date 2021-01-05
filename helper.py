'''
Helper functions.
'''

import os
import shutil


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