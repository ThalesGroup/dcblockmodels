"""Function to clean all temporary files from the repo"""

import shutil
import os


DEBUG_DIR = './dcblockmodels/model_debug_output/'
NOTEBOOK_DIR = './notebooks/'


def delete_dir(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def clean():
    if os.path.exists(DEBUG_DIR):
        for file in os.listdir(DEBUG_DIR):
            shutil.rmtree(DEBUG_DIR + file)
        delete_dir(DEBUG_DIR)

    for file in os.listdir(NOTEBOOK_DIR):
        ext = file.split('.')[-1]
        if ext in ['npy', 'npz']:
            os.remove(NOTEBOOK_DIR + file)

    delete_dir('./datasets/classic')
    delete_dir('./saved_models')

    delete_file('sparsebm.log')
    delete_file(f'{NOTEBOOK_DIR}sparsebm.log')
    delete_file('./dcblockmodels/sparsebm.log')

    delete_dir('.ipynb_checkpoints')
    delete_dir(f'{NOTEBOOK_DIR}/.ipynb_checkpoints')

    delete_dir('./dcblockmodels/tests/.pytest_cache')
    delete_dir('./dcblockmodels/models/.pytest_cache')
    delete_dir('.pytest_cache')

    delete_dir('./dcblockmodels/__pycache__')
    delete_dir('./dcblockmodels/models/__pycache__')
    delete_dir('./dcblockmodels/models/utils/__pycache__')
    delete_dir('./dcblockmodels/tests/__pycache__')


if __name__ == '__main__':
    clean()
