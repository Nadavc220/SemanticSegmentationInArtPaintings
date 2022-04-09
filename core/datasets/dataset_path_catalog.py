import os
from .pascal_sbd12 import PascalSbd12DataSet
from .dram import DramDataSet
from .cdriving import CDrivingDataSet


class DatasetCatalog(object):
    DATASET_DIR = "../data/"
    DATASETS = {
        "pascal_sbd_train": {
            "data_dir": "pascal_sbd",
            "data_list": "train.txt"
        },
        "pascal_sbd_val": {
            "data_dir": "pascal_sbd",
            "data_list": "val.txt"
        },
        "pascal_sbd12_train": {
            "data_dir": "pascal_sbd",
            "data_list": "train_filtered12.txt"
        },
        "pascal_sbd12_val": {
            "data_dir": "pascal_sbd",
            "data_list": "val_filtered12.txt"
        },
        "pascal_sbd12_styled_train": {
            "data_dir_content": "pascal_sbd",
            "data_list_content": "train_filtered12.txt",
            "data_dir_style": "DRAM_500",
            "data_list_style": "train.txt"
        },
        "pascal_sbd12_prestyled_train": {
            "data_dir": "pascal_sbd_styled",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_realism_train": {
            "data_dir": "pascal_sbd_styled_realism",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_realism_full_train": {
            "data_dir": "pascal_sbd_styled_realism_full",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_impressionism_train": {
            "data_dir": "pascal_sbd_styled_impressionism",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_impressionism_full_train": {
            "data_dir": "pascal_sbd_styled_impressionism_full",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_post_impressionism_train": {
            "data_dir": "pascal_sbd_styled_post_impressionism",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_post_impressionism_full_train": {
            "data_dir": "pascal_sbd_styled_post_impressionism_full",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_expressionism_train": {
            "data_dir": "pascal_sbd_styled_expressionism",
            "data_list": "train_filtered12.txt",
        },
        "pascal_sbd12_prestyled_expressionism_full_train": {
            "data_dir": "pascal_sbd_styled_expressionism_full",
            "data_list": "train_filtered12.txt",
        },

        "realart_train": {
            "data_dir": "realart",
            "data_list": "train.txt"
        },
        "realart_self_distill_train": {
            "data_dir": "./datasets",
            "data_list": "train.txt",
            "label_dir": 'realart/soft_labels/inference/realart_train'
        },
        "realart_test": {
            "data_dir": "realart",
            "data_list": "test.txt"
        },
        "dram_train": {
            "data_dir": "DRAM_500/",
            "data_list": "train.txt"
        },
        "dram_test": {
            "data_dir": "DRAM_500",
            "movement": "dram"
        },
        "dram_with_unseen_test": {
            "data_dir": "DRAM_500",
            "movement": "dram_with_unseen"
        },
        "dram_realism_train": {
            "data_dir": "DRAM_500",
            "movement": "realism"
        },
        "dram_realism_test": {
            "data_dir": "DRAM_500",
            "movement": "realism"
        },
        "dram_impressionism_train": {
            "data_dir": "DRAM_500",
            "movement": "impressionism"
        },
        "dram_impressionism_test": {
            "data_dir": "DRAM_500",
            "movement": "impressionism"
        },
        "dram_post_impressionism_train": {
            "data_dir": "DRAM_500",
            "movement": "post_impressionism"
        },
        "dram_post_impressionism_test": {
            "data_dir": "DRAM_500",
            "movement": "post_impressionism"
        },
        "dram_expressionism_train": {
            "data_dir": "DRAM_500",
            "movement": "expressionism"
        },

        "dram_expressionism_test": {
            "data_dir": "DRAM_500",
            "movement": "expressionism"
        },
        "dram_unseen_test": {
            "data_dir": "DRAM_500",
            "movement": "unseen"
        },
        "cdriving_train": {
            "data_dir": "C-Driving/train/compound/",
            "data_list": "3domains.txt"
        },
        "cdriving_val": {
            "data_dir": "C-Driving/val/compound/",
            "data_list": "3domains.txt"
        },
        "cdriving_with_open_val": {
            "data_dir": "C-Driving/val/compound",
            "data_list": "4domains.txt"
        },
        "cdriving_cloudy_train": {
            "data_dir": "C-Driving/train/compound/",
            "data_list": "cloudy.txt"
        },
        "cdriving_cloudy_val": {
            "data_dir": "C-Driving/val/compound/",
            "data_list": "cloudy.txt"
        },
        "cdriving_rainy_train": {
            "data_dir": "C-Driving/train/compound/",
            "data_list": "rainy.txt"

        },
        "cdriving_rainy_val": {
            "data_dir": "C-Driving/val/compound/",
            "data_list": "rainy.txt"
        },
        "cdriving_snowy_train": {
            "data_dir": "C-Driving/train/compound/",
            "data_list": "snowy.txt"

        },
        "cdriving_snowy_val": {
            "data_dir": "C-Driving/val/compound/",
            "data_list": "snowy.txt"
        },
        "cdriving_overcast_val": {
            "data_dir": "C-Driving/val/open/",
            "data_list": "overcast.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes=12, max_iters=None, transform=None):
        if "pascal_sbd12" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_dir"], attrs["data_list"]),
            )
            return PascalSbd12DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)

        elif "dram" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['data_dir']),
                movement=attrs['movement'])
            return DramDataSet(args["root"], args["movement"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)

        elif "cdriving" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['data_dir']),
                data_list=os.path.join(data_dir, attrs['data_dir'], attrs["data_list"]),
            )
            return CDrivingDataSet(args["root"], args["data_list"], max_iters=max_iters, split=mode, transform=transform)

        raise RuntimeError("Dataset not available: {}".format(name))