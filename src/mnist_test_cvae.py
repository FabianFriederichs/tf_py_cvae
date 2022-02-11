import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import datetime
import matplotlib.pyplot as plt
import cvae
import reconstruction_losses as rl

# # disable gpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# model settings
batch_size = 32
num_epochs = 15
latent_dim = 2
cond_dim = 1
kl_weight = 1.0

# load mnist dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
input_img_dim = (x_train.shape[1], x_train.shape[2], 1)
x_train = tf.expand_dims(tf.convert_to_tensor(x_train.astype('float32') / 255.0), axis = -1)
x_test = tf.expand_dims(tf.convert_to_tensor(x_test.astype('float32') / 255.0), axis = -1)
y_train = tf.convert_to_tensor(y_train.astype('float32'))
y_test = tf.convert_to_tensor(y_test.astype('float32'))

# [CVAE TEST]: define encoder and decoder models
# encoder
print("Creating encoder model...")
encoder_input = tf.keras.layers.Input(shape=input_img_dim, dtype = 'float32')
encoder_cond = tf.keras.layers.Input(shape=(cond_dim,), dtype = 'float32')
enc_tmp = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_input)
enc_tmp = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(enc_tmp)
enc_tmp = tf.keras.layers.Flatten()(enc_tmp)
enc_tmp = tf.keras.layers.Concatenate(axis = 1)([enc_tmp, encoder_cond])
enc_tmp = tf.keras.layers.Dense(4 * latent_dim, activation = 'relu')(enc_tmp)
enc_tmp = tf.keras.layers.Dense(2 * latent_dim, activation = 'linear', kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05))(enc_tmp)
encoder = tf.keras.Model([encoder_input, encoder_cond], enc_tmp, name='encoder')
encoder.summary()
# decoder
print("Creating decoder model...")
decoder_input = tf.keras.layers.Input(shape=(latent_dim,), dtype = 'float32')
decoder_cond = tf.keras.layers.Input(shape=(cond_dim,), dtype = 'float32')
dec_tmp = tf.keras.layers.Concatenate(axis = 1)([decoder_input, decoder_cond])
dec_tmp = tf.keras.layers.Dense(2 * latent_dim, activation = 'relu')(dec_tmp)
dec_tmp = tf.keras.layers.Dense(4 * latent_dim, activation = 'relu')(dec_tmp)
dec_tmp = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(dec_tmp)
dec_tmp = tf.keras.layers.Reshape((7, 7, 64))(dec_tmp)
dec_tmp = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(dec_tmp)
dec_tmp = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(dec_tmp)
dec_tmp = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='linear')(dec_tmp)
decoder = tf.keras.Model([decoder_input, decoder_cond], dec_tmp, name='decoder')
decoder.summary()
# construct CVAE model
print("Creating CVAE model...")
cvae = cvae.ConditionalVariationalAutoencoder(encoder, decoder, latent_dim, cond_dim, beta = kl_weight, use_analytical_kl=True)
print("Compiling...")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
cvae.compile(optimizer=optimizer, loss=rl.NllStandardGaussianLoss())
print("Done.")
# [CVAE TEST]: Do training
# log path for CVAE training
cvae_log_path = "./logs/cvae/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cvae_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cvae_log_path, histogram_freq=1)
print("Training CVAE model...")
cvae.fit((x_train, y_train), x_train, validation_data = ((x_test, y_test), x_test), epochs=num_epochs, batch_size=batch_size, shuffle=False, callbacks = [cvae_tensorboard_callback])

def plot_latent_images(model, n, cond, digit_size=28, decode_from_logits = False):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      c = np.array([[cond]])
      if decode_from_logits:
        x_decoded = tf.math.sigmoid(model.decode((z, c)))
      else:
        x_decoded = model.decode((z, c))
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.show()

dfl = False
plot_latent_images(cvae, 10, 0.0, 28, dfl)
plot_latent_images(cvae, 10, 1.0, 28, dfl)
plot_latent_images(cvae, 10, 2.0, 28, dfl)
plot_latent_images(cvae, 10, 3.0, 28, dfl)
plot_latent_images(cvae, 10, 4.0, 28, dfl)
plot_latent_images(cvae, 10, 5.0, 28, dfl)
plot_latent_images(cvae, 10, 6.0, 28, dfl)
plot_latent_images(cvae, 10, 7.0, 28, dfl)
plot_latent_images(cvae, 10, 8.0, 28, dfl)
plot_latent_images(cvae, 10, 9.0, 28, dfl)