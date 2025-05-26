import numpy as np                      # Biblioteca para operações com arrays
import tensorflow as tf                 # Biblioteca principal de deep learning (Framework)
from tensorflow import keras            # API de alto nível do TensorFlow para redes neurais
from tensorflow.keras import layers     # Camadas (Layers) do Keras, para construção de modelos 

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()   # Carrega imagens de treino e teste do MNIST (imagens 28x28 escalas de cinza)
x_train = x_train.astype("float32") / 255.                     # Normaliza os pixels de 0-255 para 0-1 (facilita os calculos da rede)
x_train = np.expand_dims(x_train, -1)                          # Adiciona dimensão do canal pois a rede espera um entradas 4D (fica [N, 28, 28, 1] ou [batch, x, y, 1]) 
x_test = x_test.astype("float32") / 255.                       # Normaliza os pixels de 0-255 para 0-1 (facilita os calculos da rede)
x_test = np.expand_dims(x_test, -1)                            # Adiciona dimensão do canal pois a rede espera um entradas 4D (fica [N, 28, 28, 1] ou [batch, x, y, 1])

latent_dim = 16            
batch_size = 128          
epochs = 1               

def build_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(28, 28, 1))        # Define entrada do Encoder Convolucional (fica [28x28x1] ou [x,y,channel])

    x = layers.Conv2D(
        32,                           # 32 filtros                                           # 1ª Camada Convolucional
        3,                            # Kernel 3x3                                           # 1ª Camada Convolucional
        activation="relu",            # Função de ativação ReLU                              # 1ª Camada Convolucional
        strides=2,                    # Passo (Passo 2 reduz dimensão pela metade)           # 1ª Camada Convolucional
        padding="same"                # Matem as dimensões definida pelo stride              # 1ª Camada Convolucional
        )(encoder_inputs)             # Variável de entrada da camada                        # 1ª Camada Convolucional

    x = layers.Conv2D(
        64,                           # 64 filtros                                           # 2ª Camada Convolucional
        3,                            # Kernel 3x3                                           # 2ª Camada Convolucional
        activation="relu",            # Função de ativação ReLU                              # 2ª Camada Convolucional
        strides=2,                    # Passo (Passo 2 reduz dimensão pela metade)           # 2ª Camada Convolucional
        padding="same"                # Matem as dimensões definida pelo stride              # 2ª Camada Convolucional
        )(x)                          # variavel de entrada da camada                        # 2ª Camada Convolucional

    x = layers.Flatten()(x)           # "Achata" a saída do layer em um vetor 1D para conectar na camada densa

    x = layers.Dense(
        128,                          # Camada densa (fully conected) 128 neuros             # Camada densa intermediária
        activation="relu"             # Função de ativação ReLU                              # Camada densa intermediária
        )(x)                          # Variável de entrada da camada                        # Camada densa intermediária

    z_mean = layers.Dense(
        latent_dim,                   # Saída da camada na dimensão latente                  # Saída: média das posições no espaço latente
        name="z_mean"                 # Nome da saída                                        # Saída: média das posições no espaço latente
        )(x)                          # Variável de entrada da camada                        # Saída: média das posições no espaço latente

    z_log_var = layers.Dense(
        latent_dim,                   # Saída da camada na dimensão latente                  # Saída: log da variância no espaço latente
        name="z_log_var"              # Nome da camada                                       # Saída: log da variância no espaço latente
        )(x)                          # Variável de entrada da camada                        # Saída: log da variância no espaço latente

    @keras.utils.register_keras_serializable()             # Permite salvar/carregar a função customizada junto ao keras 

    def sampling(args):                                                                             # Reparametrization Trick 
        z_mean, z_log_var = args                           # Importa média e log variância          # Reparametrization Trick
        batch = tf.shape(z_mean)[0]                        # Extrai tamanho lote (batch)            # Reparametrization Trick
        dim = tf.shape(z_mean)[1]                          # Extrai dimensão espaço latente         # Reparametrization Trick 
        epsilon = tf.random.normal(shape=(batch, dim))     # Gera ruído aleatório ∈ por batch       # Reparametrization Trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # retorna z = 𝝁 + 𝝈 ⋅ ∈                  # Reparametrization Trick
    z = layers.Lambda(                                                                              # Reparametrization Trick
        sampling,                     # Função "sampling" retorna z de forma diferenciavel          # Reparametrization Trick
        output_shape=(latent_dim,),   # Informa ao keras o formato de saída da função               # Reparametrization Trick
        name="z"                      # Nome da camada                                              # Reparametrization Trick
        )([z_mean, z_log_var])        # Informa ao keras entrada da função                          # Reparametrization Trick

    encoder = keras.Model(
        encoder_inputs,               # Cria modelo do encoder e define variavel de entrada
        [z_mean,                      # Retorna média
        z_log_var,                    # Retorna Log da variância 
        z],                           # Retorna vetor z amostrado
        name="encoder")               # Nomeia o encoder
    return encoder

def build_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))                    # Define entrada do Decoder Convolucional (vetor latente)

    x = layers.Dense   (7 * 7 * 64,                                     # Cada vetor latente será tranformado em 3136 valores
                        activation="relu"                               # Função de ativação ReLU
                        )(latent_inputs)                                # Variável de entrada da camada

    x = layers.Reshape ((7, 7, 64)                                      # Reorganiza os 3136 valores em um tensor 3D [7, 7, 64]
                        )(x)                                            # Variável de entrada da camada

    x = layers.Conv2DTranspose(64,                                      # 64 filtros                                       # 1ª Camada Deconvolucional
                                3,                                      # Kernel 3x3                                       # 1ª Camada Deconvolucional
                                activation="relu",                      # Função de ativação ReLU                          # 1ª Camada Deconvolucional
                                strides=2,                              # Passo (Passo 2 dobra a dimensão)                 # 1ª Camada Deconvolucional
                                padding="same"                          # Matem as dimensões definida pelo stride          # 1ª Camada Deconvolucional
                                )(x)                                    # variavel de entrada da camada                    # 1ª Camada Deconvolucional

    x = layers.Conv2DTranspose(32,                                      # 32 filtros                                       # 2ª Camada Deconvolucional
                                3,                                      # Kernel 3x3                                       # 2ª Camada Deconvolucional
                                activation="relu",                      # Função de ativação ReLU                          # 2ª Camada Deconvolucional
                                strides=2,                              # Passo (Passo 2 dobra a dimensão)                 # 2ª Camada Deconvolucional
                                padding="same"                          # Matem as dimensões definida pelo stride          # 2ª Camada Deconvolucional
                                )(x)                                    # variavel de entrada da camada                    # 2ª Camada Deconvolucional              

    decoder_outputs = layers.Conv2DTranspose   (1,                      # 1 filtro                                         # Camada Deconvolucional de saída
                                                3,                      # Kernel 3x3                                       # Camada Deconvolucional de saída
                                                activation="sigmoid",   # Função de ativação                               # Camada Deconvolucional de saída
                                                strides=1,              # Passo                                            # Camada Deconvolucional de saída
                                                padding="same"          # Matem a dimensão igual a entrada se stride = 1   # Camada Deconvolucional de saída
                                                )(x)                    # variavel de entrada da camada                    # Camada Deconvolucional de saída

    decoder = keras.Model  (latent_inputs,                              # Cria modelo do decoder e define variavel de entrada
                            decoder_outputs,                            # Define variavel da imagem reconstruida
                            name="decoder")                             # Nomeia o decoder
    return decoder

encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)

class VAE(keras.Model):   # Cria subclasse do modelo VAE para customização no keras

    def __init__(self, encoder, decoder, **kwargs):    # Construtor da classe, define seus argumentos  - **kwargs permite inviar mais argumentos alem dos obrigatórios
        super().__init__(**kwargs)                     # Chama o construtor da subclasse (obrigatório) - **kwargs permite inviar mais argumentos alem dos obrigatórios
        self.encoder = encoder                         # Salva encoder como atributo de objeto 
        self.decoder = decoder                         # Salva decoder como atributo de objeto

    def call(self, inputs):                            # Define oque acontece quando chamamos a classe VAE()            # Forward pass
        z_mean, z_log_var, z = self.encoder(inputs)    # Define a entrada e as saídas do encoder                        # Forward pass
        return self.decoder(z)                         # detorna o decoder(z) que é a imagem reconstruída               # Forward pass

    def train_step(self, data):                        # Passo de treinamento customizado
        if isinstance(data, tuple):                    # Ignora os rótulos do das imagens do dataset se houver
            data = data[0]                             # Ignora os rótulos do das imagens do dataset se houver

        with tf.GradientTape() as tape:                # Abre o escopo para registar as operações 0     
            z_mean, z_log_var, z = self.encoder(data)  # A entrada passa pelo encoder retornado as variaveis no espaço latente         # Forward pass
            reconstruction = self.decoder(z)           # As variaveis no espaço latente passam pelo encoder e reconstroem a imagem     # Forward pass

            reconstruction_loss = tf.reduce_mean(                                    # Faz a média sobre todas as imagens do batch
                                                tf.reduce_sum(                       # Soma a entropia cruzada sobre as dimensões espaciais (28x28)
                                                keras.losses.binary_crossentropy(    # Aplica a função de entropia cruzada
                                                data,                                # Imagem de entrada
                                                reconstruction),                     # Imagem reconstruída
                                                axis=(1, 2)))                        # Soma a entropia cruzada sobre as dimensões espaciais (28x28)
            
            kl_loss = -0.5 * tf.reduce_mean(                                                         # Faz a média sobre todas as imagens do batch 
                                            tf.reduce_sum(                                           # Soma os termos de cada dimensão latente
                                            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),   # 1 + log(σ²​)−μ²​−σ²​       
                                            axis=1))                                                 # Soma os termos de cada dimensão latente
            
            total_loss = reconstruction_loss + kl_loss                                               # Soma as perdas

        grads = tape.gradient(           # Calcula os gradientes da função de perda       # Backpropagation
                total_loss,              # Entre a função de perda                        # Backpropagation
                self.trainable_weights)  # E os pesos de treinamento                      # Backpropagation
        
        self.optimizer.apply_gradients(                 # Otimizador do modelo "adam"                           # Wnovo = Wantigo − η ⋅ ∇w loss
                zip(grads, self.trainable_weights))     # Combina o gradiente com seu respectivo peso           # W = peso 
                                                                                                                # η = taxa de aprendizado (depende do modelo)
                                                                                                                # ∇w loss = gradiente da perda em relaçãao a W 
        
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}             # Retorna métricas

def dummy_loss(y_true, y_pred): return 0.0    # Função de loss dummy (necessária para o compile)
    
vae = VAE(encoder, decoder)                     # Inicia s subclasse VAE
vae.compile(optimizer="adam",   # Define o modelo de otimizador "adam" 
            loss=dummy_loss)    # Funçao loss já é aplicada no train, sem funçao aqui

vae.fit(                                 # Inicia treinamento do modelo                                    #                        
    x_train,                             # Dataset de treinamento                                          #
    epochs = epochs,                     # Número de treinamentos                                          #
    batch_size = batch_size,             # Lotes por treinamento                                           #
    validation_data=(x_test, x_test))    # Validação (mesmo processo sem atualização de pesos)             #

encoder.save("CVAE_encoder_train.h5")  # Salva encoder treinado
decoder.save("CVAE_decoder_train.h5")  # Salva decoder treinado
