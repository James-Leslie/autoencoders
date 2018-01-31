import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


# hyperparameters
batch_size = 100
original_dim = 38
latent_dim = 2
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0

# encoder network, maps inputs to our latent space
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# we can use these parameters to sample new similar points from the latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network, maps these sampled latent points back to reconstructed points
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

# train the VAE on our input data
df = pd.read_csv("training_set.csv", low_memory=False, header=1, index_col=0)

overlay = 'Failed'

x_data = df.iloc[:, :-7].values.astype('float32')
y_data = df[overlay].values.astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.20,
                                                    random_state=42)

# vae.fit(x_train,
#         shuffle=True,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=(x_test, None))
#
# # save model and test data
# encoder = Model(x, z_mean)
# encoder.save('encoder.h5')

x_pickle = open("test_x.pickle", "wb")
pkl.dump(x_test, x_pickle)
x_pickle.close()
y_pickle = open("test_y.pickle", "wb")
pkl.dump(y_test, y_pickle)
y_pickle.close()
