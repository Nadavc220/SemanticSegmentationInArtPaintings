import argparse
import os
import logging

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger

NUM_WORKERS = 4


def knn_weights(dataset_weights, name, datasets):
    weights_dict = dataset_weights[name]
    weights = []
    for dataset in datasets:
        weights.append(weights_dict[dataset])
    if sum(weights) <= 1 - 0.00001:
        print(sum(weights))
    return weights


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label, flip=True):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        output = classifier(feature_extractor(image))
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)


def test(cfg, saveres):
    logger = logging.getLogger("FADA.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)
    
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    torch.cuda.empty_cache()
    output_folder = '.'

    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        mkdir(output_folder)

    feature_extractors = []
    classifiers = []
    for i in range(len(cfg.MODEL.MULTI_TEST_WEIGHTS)):
        feature_extractor = build_feature_extractor(cfg)
        feature_extractor.to(device)
        classifier = build_classifier(cfg)
        classifier.to(device)

        # load current checkpoint
        curr_resume = cfg.MODEL.MULTI_TEST_WEIGHTS[i]
        logger.info("Loading checkpoint from {}".format(curr_resume))
        checkpoint = torch.load(curr_resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        feature_extractor.eval()
        classifier.eval()

        feature_extractors.append(feature_extractor)
        classifiers.append(classifier)

    dataset_weights = torch.load(cfg.DATASETS.WEIGHT_VEC_DICT_PATH)

    # Build dataset
    test_data = build_dataset(cfg, mode='test', is_source=False)
    test_loader = DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, sampler=None)

    print("Testing Dataset: {}".format(cfg.DATASETS.TEST))
    mkdir(os.path.join(output_folder, 'masks'))
    mkdir(os.path.join(output_folder, 'masks_on_images'))
    mkdir(os.path.join(output_folder, 'masks_id'))

    for i, batch in enumerate(tqdm(test_loader)):
        x, y, name, path = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()
        pred = torch.tensor(np.zeros((1, 12, y[0].shape[0], y[0].shape[1]))).to(device)

        for i in range(len(cfg.MODEL.MULTI_TEST_WEIGHTS)):
            weights = knn_weights(dataset_weights, name[0], cfg.DATASETS.MULTI_TEST_WEIGHTS_DATASETS)
            out = inference(feature_extractors[i], classifiers[i], x, y, flip=False)
            pred += out.detach() * weights[i]

        output = pred.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        if saveres:
            pred = pred.cpu().numpy().squeeze()
            pred_max = np.max(pred, 0)
            pred = pred.argmax(0)

            mask = get_color_pallete(pred, "pascal12")
            mask_filename = name[0] if len(name[0].split("/"))<2 else name[0].split("/")[1]
            mask.save(os.path.join(output_folder, 'masks', mask_filename + '.png'))

            id_pred = Image.fromarray(pred.astype(np.int8))
            id_pred.save(os.path.join(output_folder, 'masks_id', mask_filename + '.png'))

            np_mask = np.array(mask.convert('RGB'))
            im = np.array(Image.open(path[0]).convert('RGB'))

            added_image = cv2.addWeighted(im, 0.35, np_mask, 0.65, 0)
            cv2.imwrite(os.path.join(output_folder, 'masks_on_images', mask_filename + '.png'), added_image[:, :, ::-1])
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--saveres', action="store_true",
                        help='save the result')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    path, model = os.path.split(cfg.resume)
    save_dir = os.path.join(path, os.path.splitext(model)[0])
    if save_dir:
        mkdir(save_dir)
    logger = setup_logger("FADA", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, args.saveres)


if __name__ == "__main__":
    main()

