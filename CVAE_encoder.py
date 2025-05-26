import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf

# Defina a função sampling igual ao treinamento
@keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Carregar encoder treinado
#encoder = keras.models.load_model("CVAE_encoder_train.h5", compile=False)
encoder = keras.models.load_model("CVAE_encoder_train.h5", compile=False)

# Carregar imagem de teste
img = Image.open("digit_7.png").convert("L").resize((28, 28))
img_arr = np.array(img).astype("float32") / 255.
img_arr = np.expand_dims(img_arr, axis=(0, -1))

# Codificar (comprimir)
z_mean, z_log_var, z = encoder.predict(img_arr)
np.save("digit_7_compressed.npy", z)