# Vision Transformer For Image Quality Perception Of Individual Observers

### Description
This project presents individual image quality assessment using Vision Transformer (ViT). Building on previous research on predicting quality ratings of compressed images using CNN models, this work aims to develop a model for human perception of image quality using ViT. The model presented in the paper "Vision Transformer for Small-Size Datasets" has been incorporated into this work for finer quality assessment. In addition, visualization of the Vision Transformer's attention mechanism will provide better insight into the human visual system.

## Visuals
Images

## Requirements
- Linux and Windows are supported, but I recommend Linux for performance and compatibility reasons.
- NVIDIA GPU. I have done all testing and development using RTX 3090 GPU.
- 64-bit Python 3.10.9 and PyTorch 2.0.1 (or later). See [pytorch.org](https://pytorch.org) for PyTorch install instructions.
- CUDA toolkit 11.8 or later.

## Installation
```
git clone https://mygit.th-deg.de/mg06201/vision-transformer-for-image-quality-perception-of-individual-observers.git
cd vision-transformer-for-image-quality-perception-of-individual-observers
conda env create -f environment.yml
conda activate vit_iqa
```

## Usage
```
jupyter notebook
```

## Support
For comments, questions and help I can be reached at the following email adress: max.geissler@stud.th-deg.de

## ToDo's
- [ ] Integrate pretrained ViT-Model
- [ ] Find optimal parameter values
- [ ] Train ViT on all distored images
- [ ] Train ViT on idividuals
- [ ] Visualize Attention-Scores
- [ ] Compare Attention-Scores of ViT with GRADCAM of JPEGResNet50
***
- [x] Write Training Script
- [x] Convert Matlab rating playlists to csv
- [x] Visualize Confusion-Map
- [x] Visualize dataset distribution
- [x] Visualize training results

## Acknowledgment
My appreciation goes to
- Marcus Barkowsky, my dedicated professor and mentor.
- Pavel Majer, whose preliminary work made individual datasets available.
- lucidrains, for their implementation of the Vision Transformer for small datasets.
