import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle as pkl


encoder = load_model('encoder.h5')
x_load = open("test_x.pickle", "rb")
x_test = pkl.load(x_load)
y_load = open("test_y.pickle", "rb")
y_test = pkl.load(y_load)

x_test_encoded = encoder.predict(x_test, batch_size=100)

# display a 2D plot of the policies in the latent space
plt.figure(figsize=(6, 6))
plt.hexbin(x_test_encoded[:, 0],
           x_test_encoded[:, 1],
           C=y_test,
           cmap='rainbow',
           linewidths=0.2,
           gridsize=200)
plt.colorbar()
plt. ylim((-3, 3))
plt. xlim((-3, 3))
# plt.savefig('Failed_overlay.png', dpi=300)

# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt. ylim((-3, 3))
# plt. xlim((-3, 3))
plt.show()
