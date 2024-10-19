# U-Net with Temporal Attention Encoder (UTAE) for Cell Segmentation

## The project:
This project consists in an implementation of the U-Net with Temporal Attention Encoder (UTAE) model to evaluate the efficiency of temporal attention for budding yeast cells segmentation. More specifically, the task is to predict a mask that segment cells given a lab microscopic image, the model therefore being used for binary classification in our case.

Images were provided by [Laboratory of the Physics of Biological Systems](https://www.epfl.ch/labs/lpbs/), with which we collaborated for the project.

For extensive information about the project motivation, implementation and results, see the project report available in 📖 [project_report.pdf](project_report.pdf)

## Repository Structure

📂 [dtsub/train_input](dtsub/train_input/) and 📂 [dtsub/train_mask](dtsub/train_mask/) - Training set of 807 images & their corresponding masks:
    - Image size is 256p x 256p
    - Input images are RGB
    - Mask images are in binary. **0 if the pixel belong to background and 1 if the pixel belong to a budding yeast cell.**

 📂 [dtsub/val_input](dtsub/val_input/)` and 📂 [dtsub/val_mask](dtsub/val_mask/) - Validation set of 121 images & their corresponding masks:
    - Same image characteristics as the training set.


**All files labeled with an asterisk contain code directly taken from the "Panoptic Segmentation Of Satellite Image Time Series With Convolutional Temporal Attention Network" research paper.**

- 📂 [image_preprocessing](image_preprocessing/): 
    -  📄 [split_data_crop_set_3.py](image_preprocessing/split_data_crop_set_3.py) & 📄 [data_augmentation.py](image_preprocessing/data_augmentation.py) : Two python files generating the images/data used by our model for training. 
- 📂 [model](model/) : Contains the python files used for the construction of UTAE model, the dataset's loading and the metrics definition & calculation.
    - 📄 [dataset.py](model/dataset.py): Loads the dataset properly.
    - 📄 [metrics.py*](model/metrics.py): Defines classes of the different metrics.
    - 📄 [mIoU.py*](model/miou.py): Handles the computation of the confusion matrix and the mIoU metric, using the previous file.
    - 📄 [model.py*](model/model.py): Core implementation of the convolutional layers & blocks used in the UTAE model.
    - 📄 [ltae.py*](model/ltae.py): Implementation of Lightweight Temporal Attention Encoder (L-TAE) for image time series and the Multi-Head Attention module.
    - 📄 [positional_encoding.py*](model/positional_encoding.py) : Implementation of the positional encoder used in the model. 
    - 📄 [weight_init.py*](model/weight_init.py): Initializes the model's parameters 
    - 📄 [utils.py](model/utils.py): Helps with dataset loading and image/results saving.

- 📄 [train_params.py](train_params.py): Hyperparameters tuning of our UTAE model, storing the results (log + importable model parameters) in a newly created 📂 [results_augmented](results_augmented/).

- 📄 [train_argparse.py](train_argparse.py): Tunes a train UTAE model and saves predicted images in a newly created 📂 [result_augmented](result_augmented/) folder.

- 📂 [plot_graph_result](plot_graph_result/): 
    -  📄 [plot_results_ipynb](plot_graph_result/plot_results.ipynb) : Plots the results using the training from 📄 [train_params.py](train_params.py).


## Prerequisites
Latest versions of :
- pyTorch
- Tensorflow
- Tqdm
- Albumentations

## Run Instructions 

Data is originally stored as a tiff file in a 📂 [cropped](cropped/) folder. To get access to the final dataset, you would be expected to run image_preprocessing/split_data_crop_set_3.py, which extract the set of images and split them on train and test set and store them in "dtsub" folder, and image_preprocessing/data_augmentation.py, which creates more data on the training set.

```
    python image_preprocessing/split_data_crop_set_3.py
    python image_preprocessing/data_augmentation.py

```

**Here, we however directly updated the data in the 📂 [dtsub](sub/) folder, as the tiff file isn't shareable in this version of the repository.**

Our model is based on the grouping of time-series images. To reproduce our results, run the following command: 

    python train_argparse.py --epochs 100 --groupby 7 --maskpos 6

Adapt the command accordingly for the epochs, groupby, and maskpos arguments of your linking.

## Results 

In our case, we used 📄 [train_params.py](train_params.py) to find the best groupby value and mask position to predict. We used this on augmented data, and found that `groupby`=7 and `maskpos`=6 yielded us the best results, evaluted on 100 epochs. The other hyperparameters used are indicated as default ones in the 📄 [train_argparse.py](train_argparse.py) file. The maskpos is unfortunately highly suceptible to stochastic noise, yielding non-absolute results

## Reference Paper:

The U-TAE model and the code was adapted from the following paper :

Vivien Sainte Fare Garnot and Loic Landrieu, "[Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks](https://arxiv.org/abs/2104.05495)," ICCV, 2021.


## Team:

  - Yassine Jamaa : yassine.jamaa@epfl.ch
  - Virginie Garnier : virginie.garnier@epfl.ch 
  - Romain Rochepeau : romain.rochepeau@epfl.ch