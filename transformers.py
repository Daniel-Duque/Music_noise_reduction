# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:13:33 2023

@author: Sebastian Pineda Daniel Duque
"""

import librosa
import librosa.display
import IPython.display as ipd

import numpy as np
import os

import os, shutil
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.layers import Dense,Flatten,Reshape,InputLayer
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix
from math import ceil
#import mlflow
#import mlflow.tensorflow
from PIL import Image
import re
from tensorflow.keras.utils import Sequence
import numpy as np   

def create_spectrogram(file_path):
    audio_array, sample_rate= librosa.load(file_path)
    spec = librosa.feature.melspectrogram(y=audio_array,
                                    sr=sample_rate, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0,
                                    n_mels=12)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    return spec,sample_rate
 

def reverse_spectrogram(log_spec,sample_rate, output_path):
    #reversed_log=librosa.db_to_power(log_spec)
    # step3 converting mel-spectrogrma back to wav file
    res = librosa.feature.inverse.mel_to_audio(log_spec, 
                                        sr=sample_rate, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=1.0, 
                                        n_iter=32)

    # step4 - save it as a wav file
    import soundfile as sf
    sf.write(output_path, res, sample_rate)
    
    

def create_spec_from_dir(dir_path,top_x=2):
    #dir_path directorio o folder donde estan los wavs
    # top_x opcional cuantos archivos maximo desea usar, dejar vacio para usarlos todos
    dir = os.listdir(dir_path)
    spec_list=[]
    s_rates=[]
    for i, file in enumerate(dir):
        try:
            if i<=top_x:
                input_file = os.path.join(dir_path, file)
                ms,sr=create_spectrogram(input_file)
                num_batches=ceil(ms.shape[1]/128) if ms.shape[1]>128 else 1
                #print(ms.shape)
                ms=np.resize(ms,(ms.shape[0],128*num_batches))
                batches=np.hsplit(ms,num_batches)
                for batch in batches:
                    spec_list.append(batch)
                    s_rates.append(sr)
        except:
            print(file," file skipped")
    
    return spec_list,s_rates

def standardize_specs(clean_specs,noisy_specs):
    
    #getting max lenght of all audios
    max_y=0
    
    for i,j in zip(clean_specs,noisy_specs):
        if i.shape[1]>max_y: max_y=i.shape[1]
        if j.shape[1]>max_y: max_y=j.shape[1]
    print(max_y)
    # reshapping all spectrogram

    for index,s in enumerate(clean_specs):
        try:
            clean_specs[index]=np.resize(s,(s.shape[0],max_y))
        except Exception as e:
            print(f"skipping clean {index} {e}")
    
    clean_specs=np.array(clean_specs)
    clean_specs=clean_specs.reshape(-1,s.shape[0],max_y,1)

    for index,s in enumerate(noisy_specs):
        try:
            noisy_specs[index]=np.resize(s,(s.shape[0],max_y))
        except Exception as e:
            print(f"skipping noise {index} {e}")
            
    noisy_specs=np.array(noisy_specs)
    noisy_specs=noisy_specs.reshape(-1,s.shape[0],max_y,1)

    return clean_specs,noisy_specs

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
    
#adapted transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.05):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.rate=rate
        self.embed_dim=embed_dim
        
        
        
        
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,

        })
        return config

    
clean_specs,clean_s_rates=create_spec_from_dir(r'data\musicnet\musicnet\train_data',24000)
noisy_specs,noisy_s_rates=create_spec_from_dir(r'data\musicnet\musicnet\train_data',24000)
s_clean_specs,s_noisy_specs=standardize_specs(clean_specs,noisy_specs)    

train_gen = DataGenerator(s_clean_specs, s_clean_specs, 32)


#we build an attention model
num_heads=3
ff_dim=10
img_shape=(s_clean_specs.shape[1],s_clean_specs.shape[2])
img_shape_flatten=(img_shape[0]*img_shape[1])


mini_trans=keras.Sequential()
mini_trans.add(layers.Input(shape=img_shape))
mini_trans.add(layers.Flatten()) 
mini_trans.add(layers.Embedding(input_dim=img_shape_flatten,
                                output_dim=20))

mini_trans.add( TransformerBlock(20, num_heads, ff_dim))              
mini_trans.add(layers.Dense(1))
mini_trans.add(layers.Reshape(img_shape))    

mini_trans.compile(optimizer='adamax', loss='mse')
mini_trans.summary()

history_min_trans = mini_trans.fit(train_gen,
                    epochs=3,
                    batch_size=32
                    )

mini_trans.save(r"mini_trans.hdf5")


