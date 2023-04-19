
import tensorflow as tf
import os
import tensorflow_io as tfio
from IPython.display import Audio
#preprocess functions



general_path=r"C:/Users/usuario/Documents/GitHub/Music_noise_reduction/data/musicnet/musicnet/train_data"

for i in os.listdir(general_path):
    particular_path=os.path.join(general_path,i)
    audio=tfio.audio.AudioIOTensor(particular_path)
    audio_slice = audio[100:]
    
    # remove last dimension
    audio_tensor = tf.squeeze(audio_slice, axis=[-1])
    

        
