# U-Net with Temporal Attention Encoder (UTAE) for Cell Segmentation

## The project:
This project consisted in an implementation of the U-Net with Temporal Attention Encoder (UTAE) model to evaluate the efficiency of temporal attention for budding yeast cells segmentation. More specifically, the task is to predict a mask that segment cells given a lab microscopic image, the model therefore being used for binary classification in our case.

For extensive information about the project motivation, implementation and results, see the project report available in `project_report.pdf`

## Datasets:

 ### `dtsub/train_input` and `dtsub/train_mask` - Training set of 807 images:
 - Image size is 256p x 256p
 - Input images are RGB
 - Mask images are in binary. **0 if the pixel belong to background and 1 if the pixel belong to a budding yeast cell.**


 ### dtsub/val_input and dtsub/val_mask - Validation set of 121 images:
 - Everything as above, except the transformation performed on the images.


## Architecture
**All files labeled with an asterisk contain code directly taken from the "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network" research paper.**

- 📂 `image_preprocessing`: 
    -  `split_data_crop_set_3.py` & `data_augmentation.py` : Used to generate the data used to train the model.
- 📂 `model` :  Contains the python files used for the construction of UTAE model, the dataset's creation and the metrics definition & calculation.
    - 📜 `dataset.py`: Loads the dataset properly.
    - 📜 `metrics.py`*: Computes the confusion matrix and mIoU (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal    Attention Network")
    - 📜 `mIoU.py`*: calculate IoU accuracy (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
    - 📜 `model.py`*: implementation of UTAE model (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
    - 📜 `mltae.py`*: implementation of Lightweight Temporal Attention Encoder (L-TAE) for image time series and Multi-Head Attention module (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
    - 📜 `positional_encoding.py`*: implementation of positional encoder (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
    - 📜 `weight_init.py`*: Initializes a model's parameters (code from the article "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network")
    - 📜 `utils.py`: load the training and test set and save the predicted images.

- 📜 `train_params.py`: Hyperparameters tuning of our UTAE model, storing the results in 📂 `result_augmented`

- 📜 `train_argparse.py`: tune a train UTAE model and save predicted image on `saved_images` folder. You can modify the hyperparameter and the folder via the terminal.

- 📂 `plot_graph_result`: 
    -  📜 `plot_results.ipynb` : Plots the results (using the training from 📜 `train_params.py`). We put on the repository on `result` the result that we get for different training.
    
## UTAE Model

## Parameter selection: 


## Prerequisites
The external libraries that we use were:
    - torch and torchnet to implement deep learning model
    - tensorflow to get augmented data

## Run Instructions 

Initially, the data are stored in tiff file in the folder "cropped". To get access to our final dataset, you have to run image_preprocessing/split_data_crop_set_3.py, which extract the set of images and split them on train and test set and store them in "dtsub" folder, and image_preprocessing/data_augmentation.py, which creates more data on the training set.

Our model is based on the grouping of time-series images. In our case, we used `train_params.py` to find the best groupby value and mask position to predict. We used this on augmented data, and found that `groupb`=7 and `maskpos`=6 give us the best result, evaluted on 100 epochs. The other hyperparameters used are indicated as default ones in the `train_argparse.py` file. The maskpos is highly suceptible to stochastic noise, so it is not an absolute result. To reproduce these data, run the following command: 

    python train_argparse.py --epochs 100 --groupby 7 --maskpos 6

## Results



## Reference

    @article{garnot2021panoptic,
      title={Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks},
      author={Sainte Fare Garnot, Vivien  and Landrieu, Loic },
      journal={ICCV},
      year={2021}
    }

## Team:
  - Romain Rochepeau : romain.rochepeau@epfl.ch
  - Yassine Jamaa : yassine.jamaa@epfl.ch
  - Virginie Garnier : virginie.garnier@epfl.ch 