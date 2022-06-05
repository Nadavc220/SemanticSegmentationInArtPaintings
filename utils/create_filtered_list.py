import os
from PIL import Image
import scipy.io
import numpy as np

dataset_path = '../data/filtered_pascal'
images_path = os.path.join(dataset_path, 'images/')
labels_path = os.path.join(dataset_path, 'labels/')
splits = ['train', 'val']
valid_labels = [3, 4, 5, 8, 9, 10, 12, 13, 15, 16, 17]

for split in splits:
    with open(os.path.join(dataset_path, split + '.txt'), mode='r') as f:
        images = f.read().splitlines()
    with open(os.path.join(dataset_path, split + '_label.txt'), mode='r') as f:
        labels = f.read().splitlines()

    images_file = open(os.path.join(dataset_path, split + '_filtered12.txt'), mode='w')
    labels_file = open(os.path.join(dataset_path, split + '_filtered12_label.txt'), mode='w')

    for image_name, label_name in zip(images, labels):
        if label_name[-3:] == 'png':
            label = np.array(Image.open(os.path.join(dataset_path, 'labels', label_name)), dtype=np.float32)
        else:  # .mat
            label = scipy.io.loadmat(os.path.join(dataset_path, 'labels', label_name))["GTcls"][0]['Segmentation'][0]

        for l in valid_labels:
            if l in label:
                images_file.write(image_name + '\n')
                labels_file.write(label_name + '\n')
                break
    images_file.close()
    labels_file.close()
        


