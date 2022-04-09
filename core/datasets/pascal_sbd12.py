import os
import numpy as np
from torch.utils import data
from PIL import Image
import scipy.io


class PascalSbd12DataSet(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=12,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        with open(os.path.join(os.path.join(data_root, self.split + '_filtered12_label.txt')), "r") as f:
            lbl_lines = f.read().splitlines()

        for fname, lbl_name in zip(content, lbl_lines):
            name = fname.strip()
            lbl_name = lbl_name.strip()
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "images/%s" % name),
                    "label": os.path.join(self.data_root, "labels/%s" % lbl_name),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            0: 0,
            3: 1,
            4: 2,
            5: 3,
            8: 4,
            9: 5,
            10: 6,
            12: 7,
            13: 8,
            15: 9,
            16: 10,
            17: 11,
        }
        self.trainid2name = {
            0: 'background',
            1: 'bird',
            2: 'boat',
            3: 'bottle',
            4: 'cat',
            5: 'chair',
            6: 'cow',
            7: 'dog',
            8: 'horse',
            9: 'person',
            10: 'pottedplant',
            11: 'sheep',
        }

        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        if datafiles['label'][-3:] == 'png':
            label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
        else:  # .mat
            label = scipy.io.loadmat(datafiles["label"])["GTcls"][0]['Segmentation'][0]
        name = datafiles["name"]

        # re-assign labels to match the format of PASCAL
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, name
