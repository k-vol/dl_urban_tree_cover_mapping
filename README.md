# Deep Learning Approach for Urban Tree Cover Mapping in Cities in Germany

This repository contains the code used for the thesis project. Data preparation, training and inference can be executed using the Jupyter Notebooks. The trained deep-learning model is in `model` directory. 

## Installation

Clone this repository and use `requirements.txt` to install the necessary Python libraries in the environment. 

Python version: 3.12.3

CUDA version: 11.8

NOTE: the training pipeline is adapted to use CUDA-compliant GPU. Should it not be available, change the `device` parameter to `"cpu"`.

## Training dataset structure

The training dataset has the following structure (example for 2 cities):

```
.
└── root/
    ├── city1/
    │   ├── masks/
    │   │   ├── product_name_tile_1.tif
    │   │   ├── ...
    │   │   └── product_name_tile_n.tif
    │   └── tiles/
    │       ├── product_name_tile_1.tif
    │       ├── ...
    │       └── product_name_tile_n.tif
    └── city2/
        ├── masks/
        │   ├── product_name_tile_1.tif
        │   ├── ...
        │   └── product_name_tile_n.tif
        └── tiles/
            ├── product_name_tile_1.tif
            ├── ...
            └── product_name_tile_n.tif
```


## Inference dataset structure

The inference dataset for this project has the following structure (example for 2 cities)

```
root/
├── city1/
│   ├── cogs/
│   │   ├── product_name_band_B01.tif
│   │   ├── ...
│   │   └── product_name_band_B12.tif
│   ├── composites/
│   │   └── product_name_composite.tif
│   └── tiles/
│       ├── product_name_tile_1.tif
│       ├── ...
│       └── product_name_tile_n.tif
└── city2/
    ├── cogs/
    │   ├── product_name_band_B01.tif
    │   ├── ...
    │   └── product_name_band_B12.tif
    ├── composites/
    │   └── product_name_composite.tif
    └── tiles/
        ├── product_name_tile_1.tif
        ├── ...
        └── product_name_tile_n.tif
```

NOTE: for making inferences only the `tiles` folder is used. To decrease the dataset size `cogs` and `composites` folders can be deleted once the `tiles` folder contains the data.