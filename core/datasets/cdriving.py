import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torch.utils import data
from PIL import Image


class CDrivingDataSet(data.Dataset):
    def __init__(self, data_root, data_list, max_iters=None, crop_size=(960, 540), mean=(128, 128, 128), transform=None, ignore_label=255, split='val'):
        self.root = data_root
        self.list_path = data_list
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        # self.mean_rgb = np.array([122.67891434, 116.66876762, 104.00698793])
        self.img_ids = [i_id.strip() for i_id in open(data_list)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.split = split
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            lbl_file = img_file[:-4] + '_train_id.png'
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": name
            })

        self.transform = transform

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle"
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')

        if self.split != 'train':
            label = np.array(Image.open(datafiles["lbl"]), dtype=np.float32)
        else:
            label = image.copy()
        size = np.array(label).shape
        name = datafiles["name"]

        # resize
        # image = image.resize(self.crop_size, Image.BICUBIC)
        # label = label.resize(self.crop_size, Image.NEAREST)

        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean_rgb
        # image = image.transpose((2, 0, 1))

        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.split != 'train':
            return image, label, name

        return image, np.array(size), name


if __name__ == '__main__':
    dst = CDrivingDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()