# Music_noise_reduction

## Contents
* [Authors]
* [Introduction]
* [Dataset Description]
* [Solution Description]
* [Requirements]
* [Instalation] 
* [Useful Links]

## Authors
| Organization   | Name | Email | 
|----------|-------------|-------------|
| PUJ-Bogota | Sebastián Pineda| juanspineda@javeriana.edu.co|
| PUJ-Bogota  |  Daniel Duque | daniel_duque@javeriana.edu.co |

## Introduction
Deep Learning Experimentation for training an autoencoder model capable of removing noise features from given audios

## Dataset Description

We used the following datasets:
* MusicNet (https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)

* Microsoft Scalable noisy speech Dataset (https://github.com/microsoft/MS-SNSD)



## Solution Description 

On the src folder you will find the notebooks used for building autoencoders:

* src\autoencoder1.ipynb : autoencoders trained on pure wav sequence
* src\autoencoder2.ipynb : autoencoders trained on mel spectrograms
* src\noise_adding.ipynb : notebook for overlaying noise audio on clean audio samples
* src\transformers.py : sample code for training transformers based on mel spectrograms


## Requirements

Basic reference of which libraries and versions were used

tensorboard=2.9.1 <br>
tensorboard-data-server=0.6.1 <br>
tensorboard-plugin-wit=1.8.1<br>
tensorflow=2.11.0 <br>
tensorflow-estimator=2.9.0 <br>
tensorflow-intel=2.11.0 <br>
tensorflow-io-gcs-filesystem=0.30.0 <br>
librosa=0.10.0.post2 <br>
matplotlib-inline=0.1.6 <br>



## Instalation 

* To create enviroment on conda:

conda create --name <env> --file requirements.txt

* To create enviroment using pip

If you want a file which you can use to create a pip virtual environment (i.e. a requirements.txt in the right format) you can install pip within the conda environment, then use pip to create requirements.txt.
<br>
<br>
conda activate <env>
conda install pip
pip freeze > requirements.txt
<br>
<br>
Then use the resulting requirements.txt to create a pip virtual environment:

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt


## Some usefull links:
  https://www.tensorflow.org/io/tutorials/audio <br>
  https://towardsdatascience.com/audio-ai-isolating-instruments-from-stereo-music-using-convolutional-neural-networks-584ababf69de <br>
  https://www.kaggle.com/datasets/imsparsh/musicnet-dataset <br>
  https://www.tensorflow.org/tutorials/audio/simple_audio <br>
