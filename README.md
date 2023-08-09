# Vision Transformer For Image Quality Perception Of Individual Observers

### Description
desc

## Visuals
Images

## Requirements
- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
- NVIDIA GPU. I have done all testing and development using RTX 3090 GPU.
- 64-bit Python 3.11 and PyTorch 2.0 (or later). See [pytorch.org](https://pytorch.org) for PyTorch install instructions.
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
For comments, queries and help I can be reached at the following email adress: max.geissler@stud.th-deg.de

## ToDo's
- [ ] Integrate pretrained ViT-Model
- [ ] Find optimal parameter values
- [ ] Train ViT on all distored images
- [ ] Train ViT on idividuals
- [ ] Visualize Attention-Scores

## Acknowledgment
My appreciation goes to
- Marcus Barkowsky, my dedicated professor and mentor.
- Pavel Majer, whose preliminary work made individual datasets available.
- lucidrains, for their implementation of the Vision Transformer for small datasets.