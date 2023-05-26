
import tensorflow as tf
import os
import tensorflow_io as tfio
from IPython.display import Audio
from scipy.io.wavfile import read,write
import numpy as np
#preprocess functions



general_path=r"C:/Users/usuario/Documents/GitHub/Music_noise_reduction/data/musicnet/musicnet/train_data"

time=16000
sizes=1000



#TODO make this able to receive multiple sizes
dataset=np.empty((sizes))
for i in os.listdir(general_path):
    part_path=os.path.join(general_path,i)
    song = read(part_path)
    npsong=np.array(song[1],dtype=float)
    padding=sizes-len(npsong)
    if padding>0:
        padsong=np.pad(npsong, (0,padding), 'constant', constant_values=(0,0))
    else:
        padsong=npsong[:sizes]
    mean=-2.74658203125e-07
    padsongnorm=padsong
    dataset=np.vstack((dataset,padsongnorm))

model_rn = tf.keras.Sequential()
model_rn.add(tf.keras.layers.LSTM(100, input_shape=(sizes,1,), return_sequences=True))
model_rn.add(tf.keras.layers.LSTM(1, return_sequences=True))
opti=tf.keras.optimizers.Adam(learning_rate=0.00001)
model_rn.compile(loss="MSE",optimizer=opti)
model_rn.fit(dataset,dataset,epochs=10,validation_split=0.2)
fitted=model_rn.predict(dataset)

