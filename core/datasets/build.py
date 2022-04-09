from . import transform
from .dataset_path_catalog import DatasetCatalog

def build_base_transform(cfg, is_source):
    w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
    trans = transform.Compose([
    transform.Resize((h, w), resize_label=True),
    transform.ToTensor()])
    return trans


def build_base_transform_no_cfg(resize_shape=None):
    if resize_shape:
        trans_list = [transform.Resize((h, w), resize_label=True)]
    else:
        trans_list = []
    trans_list.append(transform.ToTensor())
    trans = transform.Compose(trans_list)
    return trans

def build_inference_transform(size, mean, std, to_bgr=False):
    w, h = size
    trans = transform.Compose([
        transform.Resize((h, w), resize_label=False),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std, to_bgr255=to_bgr)
    ])
    return trans

def build_transform(cfg, mode, is_source):
    if mode=="train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN[0]==cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
            trans_list = [transform.Resize((h, w)),] + trans_list
        else:
            trans_list = [
                transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                transform.RandomCrop(size=(h, w), pad_if_needed=True),
            ] + trans_list
        if is_source:
            trans_list = [
                 transform.ColorJitter(
                    brightness=cfg.INPUT.BRIGHTNESS,
                    contrast=cfg.INPUT.CONTRAST,
                    saturation=cfg.INPUT.SATURATION,
                    hue=cfg.INPUT.HUE,
                ),
            ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans


def build_dataset(cfg, mode='train', is_source=True, epochwise=False):
    assert mode in ['train', 'val', 'test']
    trans = build_transform(cfg, mode, is_source)
    iters = None
    if mode=='train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=trans)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=trans)
    elif mode=='val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=trans)
    elif mode=='test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=trans)
    return dataset


def build_dataset_no_cfg(cfg, dataset, mode='train', is_source=True, epochwise=False):
    assert mode in ['train', 'val', 'test']
    trans = build_transform(cfg, mode, is_source)
    iters = None
    if mode=='train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE
    dataset = DatasetCatalog.get(dataset, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=trans)
    return dataset

