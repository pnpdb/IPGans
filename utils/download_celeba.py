import os
import zipfile

working_dir = os.path.dirname(os.getcwd())
# os.system('wget -P ' + working_dir + '/data/CelebA/ https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip')

with zipfile.ZipFile(working_dir + '/data/CelebA/celeba.zip', "r") as zip_ref:
    zip_ref.extractall(working_dir + '/data/CelebA/')
