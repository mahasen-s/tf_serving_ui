import os
import numpy as np
from shutil import copyfile

# Define parameters
N_images = 100
random_seed = 100

# Define img directories
test_dir = '/home/soo00c/data/format_voc/test/'
img_dir = test_dir+'images/'
ann_dir = test_dir+'annotations/'
output_dir = '/home/soo00c/Pictures/boeing/demo/'

# Prep
os.system('rm -rf '+output_dir+'*');

# Get list of files
img_filelist = os.listdir(img_dir)
img_names = [os.path.splitext(x)[0] for x in img_filelist]

# Set random seed and sample without replacement
np.random.seed(random_seed)
rand_img_names = np.random.choice(img_names, size = N_images, replace=False)

# Copy files
for img_name in rand_img_names:
    # copy image
    copyfile(img_dir+img_name+'.jpg',output_dir+img_name+'.jpg')
    
    # copy annotation
    copyfile(ann_dir+img_name+'.xml',output_dir+img_name+'.xml')
