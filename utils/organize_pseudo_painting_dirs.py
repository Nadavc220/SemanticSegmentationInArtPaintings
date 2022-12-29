"""Use this after getting a folder of style transfered images using https://github.com/bethgelab/stylize-datasets"""
import os
from shutil import copyfile, move

DATA_DIR = '../data'
PASCAL_SBD_DIR = '../data/pascal_sbd_new'
TRAIN_IM_LIST_FILE = 'train_filtered12.txt'
TRAIN_LBL_LIST_FILE = 'train_filtered12_label.txt'
RELEVANT_MOVEMENTS = ['realism', 'impressionism', 'post_impressionism', 'expressionism']

for movement in RELEVANT_MOVEMENTS:
    new_path = os.path.join(DATA_DIR, f'pascal_sbd_styled_{movement}')
    print(f"organizing {movement} pseudo paintings in {new_path}")

    styles_image_names = list(map(lambda x: x[:-4], filter(lambda x: x.split('.')[-1] == 'jpg', os.listdir(new_path))))

    os.makedirs(os.path.join(new_path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(new_path, 'images'), exist_ok=True)

    f_im = open(os.path.join(new_path, TRAIN_IM_LIST_FILE), mode='w')
    f_lbl = open(os.path.join(new_path, TRAIN_LBL_LIST_FILE), mode='w')

    with open(os.path.join(PASCAL_SBD_DIR, TRAIN_IM_LIST_FILE), mode='r') as f:
        image_files = f.read().splitlines()

    count = 0
    for im_file in image_files:
        candidates = list(filter(lambda x: im_file[:-4] in x, styles_image_names))
        if len(candidates) > 1:
            print("found too many candidates for " + im_file)
            print(candidates)
            exit(1)
        new_name = candidates[0]
        if os.path.exists(os.path.join(PASCAL_SBD_DIR, 'labels', im_file[:-4] + '.png')):
            copyfile(os.path.join(PASCAL_SBD_DIR, 'labels', im_file[:-4] + '.png'), os.path.join(new_path, 'labels', new_name + '.png'))
            f_lbl.write(new_name + '.png\n')
        elif os.path.exists(os.path.join(PASCAL_SBD_DIR, 'labels', im_file[:-4] + '.mat')):
            copyfile(os.path.join(PASCAL_SBD_DIR, 'labels', im_file[:-4] + '.mat'), os.path.join(new_path, 'labels', new_name + '.mat'))
            f_lbl.write(new_name + '.mat\n')
        else:
            print("Failed to find label: " + im_file)
            exit(1)

        count += 1

    for im_name in styles_image_names:
        file_name = im_name + '.jpg'

        move(os.path.join(new_path, file_name), os.path.join(new_path, 'images', file_name))
        f_im.write(file_name + '\n')

    f_im.close()
    f_lbl.close()
    print('copied {} labels'.format(count))