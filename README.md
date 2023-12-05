# InBrainSyn

![Pipeline](./figs/pipeline.png)

This repository contains the source code for the research paper titled "*Individualized Brain MRI Synthesis from a Single Scan: Applications to Aging and Alzheimer's Disease*". You can find the paper [here](xxx).

## TODOs
- [ ] Add the link to the paper
- [X] Add the Dockerfile to run Pole Ladder
- [X] Share the pre-trained weights

## OASIS-3 Dataset
If you use this dataset (including the examples in this repo), please cite the following and refer to the corresponding [Data Use Agreement](https://www.oasis-brains.org/#access) .
- OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease.
Pamela J LaMontagne, Tammie L.S. Benzinger, John C. Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei Vlassenko, Marcus E. Raichle, Carlos Cruchaga, Daniel Marcus, 2019. medRxiv. doi: 10.1101/2019.12.13.19014902

## Pre-trained AD and HC Template Creation Models
We offer pretrained Atlas-GAN weights trained on the OASIS-3 Dataset for simulating normal aging and Alzheimer's Disease. Additionally, we provide inference scripts for extracting the learned diffeomorphic registration module and template generation module. (in `./models` and `./src`)

## Instructions
The main steps in the pipeline are summarized and demonstrated in `InBrainSyn.py` using a subject from the OASIS3 dataset, transitioning from healthy to AD. Pole ladder is used for parallel transport. If you use InBrainSyn, please also cite the following:

- Longitudinal Analysis of Image Time Series with Diffeomorphic Deformations: A Computational Framework Based on Stationary Velocity Fields.
Mehdi Hadj-Hamou, Marco Lorenzi, Nicholas Ayache, and Xavier Pennec, 2016. Frontiers in neuroscience. doi: 10.3389/fnins.2016.00236

We also provide Dockerfile in the directory `./Dockerfiles` to ease the use of this tool. Note that you need to download [Pole ladder](http://www-sop.inria.fr/teams/asclepios/software/LCClogDemons/Ladder.tar.gz) and cmake (I used [cmake-3.17.3](https://cmake.org/files/v3.17/)), and then unzip them in the same folder to successfully build the Docker image.

## Citation
```
TBA
```

## Acknowledgements:
This repository is developed based on the [Atlas-GAN](https://github.com/neel-dey/Atlas-GAN) project and makes extensive use of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library. [SynthSeg](https://github.com/BBillot/SynthSeg) version 2.0 is used to get segmentation masks. 

