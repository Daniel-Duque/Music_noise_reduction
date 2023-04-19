
import tensorflow as tf
import os
import tensorflow_io as tfio
from IPython.display import Audio

#preprocess functions



general_path=r"C:/Users/usuario/Documents/GitHub/Music_noise_reduction/data/musicnet/musicnet/train_data"

def audio_dataset(general_path):
    n=0
    for i in os.listdir(general_path):
        n+=1
        particular_path=os.path.join(general_path,i)
            
        audio = tfio.audio.AudioIOTensor(particular_path)
        audio_slice = audio[:]

        # remove last dimension
        audio_tensor = tf.squeeze(audio_slice)
        
        print(audio_tensor.numpy())
        print(i)
        rate=audio.rate.numpy()
        
        
audio_dataset(general_path)