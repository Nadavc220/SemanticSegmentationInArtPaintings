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
After running the all instructions, your folder tree should look like this:
```bash
├── data
│   ├── pascal                                  ---> original pascal dataset
│   ├── sbd                                     ---> original sbd dataset
│   ├── pascal_sbd                              ---> pascal combined with sbd and filtered (same as paper)
│   ├── pascal_sbd_styled_realism               ---> realism pseudo-paintings
│   ├── pascal_sbd_styled_impressionism         ---> impressionism pseuso-paintings
│   ├── pascal_sbd_styled_post_impressionism    ---> post-impressionism pseudo-paintings
│   ├── pascal_sbd_styled_expressionism         ---> expressionism pseudo paintings
│   └── DRAM_500                                ---> DRAM art paintings dataset
│
├── SemanticSegmentationInArtPaintings          ---> project main directory
├── pytorch-AdaIN                               ---> AdaIN style transfer network code
└── stylize-datasets                            ---> dataset stylization repository for pseudo-paintings creation
```

- Download Data
    - Create data dir as described above (located in the same folder as your ProjectDir)
    - Download PascalVoc12 dataset [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and unzip it in data/pascal.
    - Download SBD dataset [link](http://home.bharathh.info/pubs/codes/SBD/download.html) and unzip it in data/sbd.
    - combine and filter pascal and sbd to one dataset with 12 classes as used in the paper:

    ```
    # combine
    python utils/combine_pascal_sbd.py
    # filter
    python utils/create_filtered_list.py 
    ```
    
    - Download DRAM dataset [link](https://faculty.runi.ac.il/arik/site/artseg/Dram-Dataset.html) and unzip it to data. (Rename folder from DRAM_processed to DRAM_500, sorry about that...)

- Train AdaIN style transfer networks:   
    - clone and install [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
    - create combined DRAM data folder compatible with pytorch-AdaIN interface:
        
    ```
    from main project dir:
    python utils/organize_dram_for_adain_train.py
    ```

    This will create a folder in your data dir: DRAM_for_Adain. This folders hold all of the movements images in a single folder for traninig the AdaIN network.
    Feel free to remove 'DRAM_for_Adain' after the next step, as you will not need it further.

    - Train AdaIN weights using the following command from pytorch-AdaIN project dir:

    ```
    CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --content_dir ../data/pascal_sbd/images --style_dir ../data/DRAM_for_Adain
    ```
        
 - Create a Psuedo-Paintings dataset for each movement.
    - clone and install [stylize-datasets](https://github.com/bethgelab/stylize-datasets)
    - copy and rename the decoder '.pth' weight file from pytorch-Adain/experiments/decoder2_iter_160000.pth.tar -> stylize-datasets/models/decoder.pth
    - create a data folder for each art movement compatible with stylize_datasets interface:
    
    ```
    python utils/organize_dram_for_stylization.py
    ```
    
    This will create a folder for each art movement in your data dir: DRAM_for_stylization_<movement> same as we did before for training AdaIN
    Feel free to remove these folders after the next step, as you will not them further.
    
    - for each created folder, run stylize-datasets/stylize.py script from its project page to create pseudo painting dataset as follows:

    ```
    python stylize.py --content-dir ../data/pascal_sbd/images --style-dir ../data/DRAM_for_stylization_realism --num-styles 1 --content-size 0 --style-size 0 --alpha 0.5 --output-dir ../data/pascal_sbd_styled_realism
    
    python stylize.py --content-dir ../data/pascal_sbd/images --style-dir ../data/DRAM_for_stylization_impressionism --num-styles 1 --content-size 0 --style-size 0 --alpha 0.5 --output-dir ../data/pascal_sbd_styled_impressionism
    
    python stylize.py --content-dir ../data/pascal_sbd/images --style-dir ../data/DRAM_for_stylization_post_impressionism --num-styles 1 --content-size 0 --style-size 0 --alpha 0.5 --output-dir ../data/pascal_sbd_styled_post_impressionism
    
    python stylize.py --content-dir ../data/pascal_sbd/images --style-dir ../data/DRAM_for_stylization_expressionism --num-styles 1 --content-size 0 --style-size 0 --alpha 0.5 --output-dir ../data/pascal_sbd_styled_expressionism
    ```
    
    - run the following script from the main project dir to organize the pseudo painting folders for the Semantic Segmentation in Art Paintings training:
    
    ```
    python utils/organize_pseudo_painting_dirs.py
    ```


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
