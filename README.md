# Semantic Segmentation in Art Paintings
This is the official implementation of the paper **Semantic Segmentation in Art Paintings**. EuroGraphics 2022.  
Nadav Cohen, Yael Newman, Ariel Shamir.  
[\[Arxiv\]](https://arxiv.org/abs/2203.03238) [\[EG\]](https://diglib.eg.org/handle/10.1111/cgf14473)

![teaser](figures/paper_teaser.png)

We are organizing our code so it is understandable and easy to use. Until then we invite you to read our paper.


## Updates

***06/05/2022***

`News`: DRAM dataset is now available in our project page (seem more info below). Also available are our trained checkpoints, style weights file for DRAM test set
and scripts for combining PascalVOC12 and SBD datasets and filtering the combined dataset to include the 12 classes used in the paper.  
`Next Update`: Final scripts for stylizing the dataset with a specific art style and clear instructions for train and test scripts (requirements.txt file, and command line calls). Hopfully next update will be the final one (besides bug fixes if needed).

***04/09/2022***

`News`: Uploaded train and test scripts, and everything needed to run them.  
`Next Update`: Link to DRAM dataset, trained checkpoints and scripts for PascalVOC2012+SBD preprocessing.

### Project Page + Dataset Download
Our project page is officially up: [\[Project Page\]](https://faculty.runi.ac.il/arik/site/artseg/)  
Here you can find the download links for DRAM dataset and some more info about the paper.  
`Note`: The website offers two options: RAW and Processed. The processed version is the dataset in the state we used it before training. RAW holds the dataset
as gathered from original sources without any process.

### Checkpoints
You can find our trained checkpoints in the following link: [\[checkpoints.zip\]](https://faculty.runi.ac.il/arik/site/artseg/checkpoints.zip)  
The .zip file holds checkpoints for Step1 and Step2 for each of the art movements used for training: Realism, Impressionism, Post-Impressionism and Expressionism.  
To compose the networks as described in the paper unzip the checkpoints.zip file to the project folder and then run ./test_multi_weighted.py together with the style weights file (found in the style weights folder).  
The multi-test config should already be set to use the checkpoints and style weights.

## Usage
### Env Installation
1) Install Python 3.8.5
2) Create a pip virtual env using python -m venv <path_to_env>
3) run 'pip install -r requirements.txt' from the project folder.
4) activate pip env: 'source <path_to_env>/bin/activate.

### Data Preparation
```bash
├── data
│   ├── pascal
│   ├── sbd
│   └── DRAM_500
└── ProjectDir
```

1) Create data dir as described above (located in the same folder as your ProjectDir)
2) Download PascalVoc12 dataset [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and unzip it in data/pascal.
3) Download SBD dataset [link](http://home.bharathh.info/pubs/codes/SBD/download.html) and unzip it in data/sbd.
4) combine and filter pascal and sbd to one dataset with 12 classes as used in the paper:
```
# combine
python utils/combine_pascal_sbd.py
# filter
python utils/create_filtered_list.py 
```
5) Download DRAM dataset [link](https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html) and unzip it to data. (Rename folder from DRAM_processed to DRAM_500, sorry about that...)


### Acknowledge
Some of our code is adapted from [FADA](https://github.com/JDAI-CV/FADA) and [Permuted AdaIN](https://github.com/onuriel/PermutedAdaIN). We thank them for their excellent research and for sharing their repositories.

# Citation
If you found this project helpful in your research, please consider citing our paper.
```
@article {ArtSeg-22,
    journal = {Computer Graphics Forum},
    title = {{Semantic Segmentation in Art Paintings}},
    author = {Cohen, Nadav and Newman, Yael and Shamir, Ariel},
    year = {2022},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.14473}
}
```
