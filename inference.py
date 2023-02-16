import argparse
import os
import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from core.configs import cfg
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, get_color_pallete
from core.datasets import build_transform

from dram_style_weights.models.gram_embedder import GramEmbedder

KN = 500


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(img, feature_extractor, classifier,  orig_shape):
    with torch.no_grad():
        output = classifier(feature_extractor(img))

    output = F.interpolate(output, size=orig_shape, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    output = output[0]
    return output.unsqueeze(dim=0)


def infer_img(img, orig_shape, feature_extractors, classifiers, gram_embedder, dram_train_embeds, pca_f, device, k_neigh=500):
    print(f"Finding train set KNN... (k={KN})")
    # find gram representation of img
    img_gram_vec = gram_embedder(img)
    img_embed = pca_f.transform(img_gram_vec.cpu().numpy()[np.newaxis, :])

    # Find KNN gram vectors
    end_indices = [0]
    dists = []
    for i, dataset_name in enumerate(cfg.DATASETS.MULTI_TEST_WEIGHTS_DATASETS):
        end_indices.append(len(dram_train_embeds[dataset_name]) + end_indices[i])
        dists.append(cdist(img_embed, dram_train_embeds[dataset_name]))
    dists = np.concatenate(dists, axis=1)[0]
    knn_indices = np.argsort(dists)[:KN]

    # calculate KNN for net-weights
    weights = []
    for i in range(1, len(end_indices)):
        count = np.count_nonzero(np.logical_and(end_indices[i - 1] <= knn_indices, knn_indices < end_indices[i]))
        weights.append(count / len(knn_indices))
    print(f"weights: {weights}")

    # get image prediction of each model
    print("merging predictions...")
    pred = torch.tensor(np.zeros((1, 12, orig_shape[0], orig_shape[1]))).to(device)
    for i in range(len(cfg.MODEL.MULTI_TEST_WEIGHTS)):
        out = inference(img, feature_extractors[i], classifiers[i], orig_shape)
        pred += out.detach() * weights[i]

    output = pred.max(1)[1].squeeze().cpu().numpy()
    return output


def main(img_paths, output_path):
    device = torch.device(cfg.MODEL.DEVICE)
    torch.cuda.empty_cache()

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
        print(f"Loading checkpoint from {curr_resume}")
        checkpoint = torch.load(curr_resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        feature_extractor.eval()
        classifier.eval()

        feature_extractors.append(feature_extractor)
        classifiers.append(classifier)

    # initialize vgg model for GRAM matrix computation
    gram_embedder = GramEmbedder().to(device).eval()
    pca_f = torch.load(cfg.MODEL.PCA)
    dram_train_embeds = torch.load(cfg.DATASETS.TRAIN_GRAM_EMBEDDINGS)

    trans = build_transform(cfg, mode='test', is_source=True)

    for img_path in img_paths:
        print(f"inferring <{img_path}>")
        img = Image.open(img_path).convert('RGB')
        orig_shape = (img.size[1], img.size[0])
        img_tensor, _ = trans(img, img)  # ignore second arg
        img_tensor = img_tensor.unsqueeze(0).to(device)

        pred = infer_img(img_tensor, orig_shape, feature_extractors, classifiers, gram_embedder, dram_train_embeds, pca_f, device, k_neigh=500)
        mask = get_color_pallete(pred, "pascal12")

        name, ext = img_path.split(os.sep)[-1].split('.')
        mask.save(os.path.join(output_path, name + '_mask.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Testing")
    parser.add_argument("-cfg", "--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("-input_path",
                        metavar="FILE",
                        help="path to an img file or a folder with images",
                        type=str)
    parser.add_argument("-output_path",
                        default=None,
                        metavar="FILE",
                        help="path to an img file or a folder with images",
                        type=str)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if os.path.isdir(args.input_path):
        img_paths = [os.path.join(args.input_path, name) for name in os.listdir(args.input_path)]
    else:
        img_paths = [args.input_path]

    output_path = args.output_path
    if output_path is None:
        output_path = './infer_outputs'
    os.makedirs(output_path, exist_ok=True)

    main(img_paths, output_path)

