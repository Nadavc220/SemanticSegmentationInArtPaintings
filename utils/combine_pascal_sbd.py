import os
from shutil import copyfile

# path to VOCdevkit/VOC2012
pascal_path = '../data/pascal/VOCdevkit/VOC2012'
# path to benchmark_RELEASE/dataset
sbd_path = '../data/sbd/benchmark_RELEASE/dataset'

output_dir = '../data/pascal_sbd'
os.makedirs(output_dir, exist_ok=True)

# handle pascal lists
train_f = open(os.path.join(pascal_path, 'ImageSets/Segmentation', 'train.txt'), mode='r')
val_f = open(os.path.join(pascal_path, 'ImageSets/Segmentation', 'val.txt'), mode='r')

pascal_train_lst = train_f.readlines()
pascal_val_lst = val_f.readlines()

train_f.close()
val_f.close()

# handle sbd lists
train_f = open(os.path.join(sbd_path, 'train.txt'), mode='r')
val_f = open(os.path.join(sbd_path, 'val.txt'), mode='r')

sbd_train_lst = train_f.readlines()
sbd_val_lst = val_f.readlines()

train_f.close()
val_f.close()

# create validation set
val_img_lst = []
val_lbl_lst = []
val_names = set()

for name in pascal_val_lst:
    val_names.add(name)
    val_img_lst.append(name[:-1] + '.jpg')
    val_lbl_lst.append(name[:-1] + '.png')

# combine lists
train_img_lst = []
train_lbl_lst = []
current_names = set()

image_loc_map = {}
for i, lst in enumerate([pascal_train_lst, sbd_train_lst, sbd_val_lst]):
    for name in lst:
        if name not in current_names and name not in val_names:
            current_names.add(name)
            train_img_lst.append(name[:-1] + '.jpg')
            image_loc_map[name[:-1] + '.jpg'] = i
            if i == 0:
                train_lbl_lst.append(name[:-1] + '.png')
            else:
                train_lbl_lst.append(name[:-1] + '.mat')


print('{} Images in train set...'.format(len(train_img_lst)))
print('{} Images in val set...'.format(len(val_img_lst)))
print('Copying data...')

src_img_pascal = os.path.join(pascal_path, 'JPEGImages')
src_img_sbd = os.path.join(sbd_path, 'img')
src_lbl_pascal = os.path.join(pascal_path, 'SegmentationClass')
src_lbl_sbd = os.path.join(sbd_path, 'cls')

trg_img_path = os.path.join(output_dir, 'images')
trg_lbl_path = os.path.join(output_dir, 'labels')

os.makedirs(trg_img_path, exist_ok=True)
os.makedirs(trg_lbl_path, exist_ok=True)


for i, file in enumerate(train_img_lst):
    if image_loc_map[file] == 0:
        copyfile(os.path.join(src_img_pascal, file), os.path.join(trg_img_path, file))
    else:
        copyfile(os.path.join(src_img_sbd, file), os.path.join(trg_img_path, file))

for file in val_img_lst:
    copyfile(os.path.join(src_img_pascal, file), os.path.join(trg_img_path, file))

for file in train_lbl_lst:
    if file[-3:] == 'png':
        copyfile(os.path.join(src_lbl_pascal, file), os.path.join(trg_lbl_path, file))
    else:
        copyfile(os.path.join(src_lbl_sbd, file), os.path.join(trg_lbl_path, file))

for file in val_lbl_lst:
    copyfile(os.path.join(src_lbl_pascal, file), os.path.join(trg_lbl_path, file))

assert len(os.listdir(os.path.join(output_dir, 'images'))) == len(train_img_lst) + len(val_img_lst)
assert len(os.listdir(os.path.join(output_dir, 'labels'))) == len(train_lbl_lst) + len(val_lbl_lst)

print('writing txt files...')
train_f = open(os.path.join(output_dir, 'train.txt'), mode='w')
for name in train_img_lst:
    train_f.write(name + '\n')
train_f.close()

val_f = open(os.path.join(output_dir, 'val.txt'), mode='w')
for name in val_img_lst:
    val_f.write(name + '\n')
val_f.close()

label_f = open(os.path.join(output_dir, 'val_label.txt'), mode='w')
for name in val_lbl_lst:
    label_f.write(name + '\n')
label_f.close()

train_label_f = open(os.path.join(output_dir, 'train_label.txt'), mode='w')
for name in train_lbl_lst:
    train_label_f.write(name + '\n')
train_label_f.close()

print('Done')


