import os
import numpy as np
import torch
from torch.utils import data
from sklearn.decomposition import KernelPCA

from core.datasets.dataset_path_catalog import DatasetCatalog
from core.datasets.build import build_base_transform_no_cfg

from gram_embedding.models.gram_embedder import GramEmbedder

MAX_VECS = 2000

N_COMPONENTS = 512
NUM_WORKERS = 0
EMBEDDINGS_FILE_NAME = 'trainset_embeddings' + str(N_COMPONENTS)
DEVICE = "cuda:0"
OUTPUT_DIR = os.path.join('gram_embedding')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RELEVANT_DATASETS = ['dram_realism_train', 'dram_impressionism_train', 'dram_post_impressionism_train', 'dram_expressionism_train']

gram_embedder = GramEmbedder().to(DEVICE).eval()

data_lengths = []
data_loaders = []
trans = build_base_transform_no_cfg()
for j, dataset in enumerate(RELEVANT_DATASETS):
    data_split = dataset.split('_')[-1]
    dataloader = DatasetCatalog.get(dataset, data_split, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )

    data_loaders.append(train_loader)

    dataset_len = len(train_loader)
    if dataset_len <= MAX_VECS:
        data_lengths.append(dataset_len)
    else:
        data_lengths.append(MAX_VECS)

curr_embeddings = {}
gram_vecs = None
for j, dataset in enumerate(RELEVANT_DATASETS):
    dataloader = data_loaders[j]
    previous_datasets_lengths = sum(data_lengths[:j])
    print("Encoding {} source images...".format(data_lengths[j]))
    for i, (im, _, _) in enumerate(dataloader):
        im = im.to(DEVICE)
        im_gram = gram_embedder(im)

        if i == 0 and j == 0:  # set size of embeddings array
            gram_vecs = np.zeros(shape=(sum(data_lengths), im_gram.size()[0]))
        gram_vecs[i + previous_datasets_lengths] = im_gram.detach().cpu().numpy()

        if i + 1 == data_lengths[j]:
            break


pca = KernelPCA(n_components=N_COMPONENTS, kernel='cosine')
if len(RELEVANT_DATASETS) > 0:
    print("Running PCA on data...")
    embedded = pca.fit_transform(gram_vecs)

    for i in range(len(RELEVANT_DATASETS)):
        previous_datasets_lengths = sum(data_lengths[:i])
        data = embedded[previous_datasets_lengths: previous_datasets_lengths + data_lengths[i]]
        curr_embeddings[RELEVANT_DATASETS[i]] = data

torch.save(pca, os.path.join(OUTPUT_DIR, 'pca_f'), pickle_protocol=4)
torch.save(curr_embeddings, os.path.join(OUTPUT_DIR, EMBEDDINGS_FILE_NAME))
