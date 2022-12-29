import os
from shutil import copyfile

source_path = '../data/DRAM_500/'
target_path = '../data/DRAM_for_Adain'
os.makedirs(target_path, exist_ok=True)
relevant_sets = ['realism', 'impressionism', 'post_impressionism', 'expressionism']

count = 0
for movement in relevant_sets:
    images_path = os.path.join(source_path, 'train', movement)
    for artist in os.listdir(images_path):
        curr_path = os.path.join(images_path, artist)
        for img_file in os.listdir(curr_path):
            new_name = '_'.join([movement, artist, img_file])
            copyfile(os.path.join(curr_path, img_file), os.path.join(target_path, new_name))
            count += 1
    print('copied {} images from {}'.format(count, movement))
    count = 0