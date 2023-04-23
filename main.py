
import tensorflow as tf
import os
import tensorflow_io as tfio
from IPython.display import Audio

#preprocess functions



general_path=r"C:/Users/usuario/Documents/GitHub/Music_noise_reduction/data/musicnet/musicnet"
time=16000


dataset=tf.keras.utils.audio_dataset_from_directory(
    general_path,
    labels='inferred',
    label_mode='int',
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=time,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False
)

model_rn = tf.keras.Sequential()
model_rn.add(tf.keras.layers.LSTM(8, input_shape=(2,5,)))


model_rn.compile(loss=)
model_rn.fit(dataset)