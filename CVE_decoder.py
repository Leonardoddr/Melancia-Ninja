import numpy as np
from tensorflow import keras
from PIL import Image

# Carregar decoder treinado
#decoder = keras.models.load_model("CVAE_decoder_train.h5", compile=False)
decoder = keras.models.load_model("CVAE_decoder_train.h5", compile=False)

# Carregar vetor comprimido
z = np.load("digit_7_compressed.npy")

# Decodificar (reconstruir)
reconstructed = decoder.predict(z)
reconstructed_img = (reconstructed[0, :, :, 0] * 255).astype("uint8")
img = Image.fromarray(reconstructed_img)
img.save("digit_7_reconstructed.png")