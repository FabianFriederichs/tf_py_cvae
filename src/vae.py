import tensorflow as tf
import numpy as np

### Python module, implementing a generic VAE

# calculates log normal pdf, given mean, log variances (assume diagonal covariance) and the dimensional axis which defaults to 1
def log_normal_pdf(x, mean, logvar):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * (
            tf.math.square(x - mean) * tf.math.exp(-logvar) + logvar + log2pi
        ),
        axis = -1
    )

# Layer for introducing a latent space sample from a normal distribution using the reparameterization trick
class NormalDistSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, logvar = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape = (batch, dim))
        return mean + tf.exp(logvar * 0.5) * eps

# Generic VAE class
class VariationalAutoEncoder(tf.keras.Model):
    # CTOR, takes encoder and decoder model + latent dimension
    def __init__(self, encoder, decoder, latent_dim, beta = 1.0, use_analytical_kl = True):
        super(VariationalAutoEncoder, self).__init__()
        # model params
        self.latent_dim = latent_dim
        self.use_analytical_kl = use_analytical_kl
        self.beta = beta
        # model parts
        self.encoder = encoder
        self.sample_latent = NormalDistSampling()
        self.decoder = decoder
        # metrics for reconstruction loss, regularization loss and total loss
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = "reconstruction_loss")
        self.regularization_loss_tracker = tf.keras.metrics.Mean(name = "regularization_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name = "total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.regularization_loss_tracker
        ]

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), 2, axis = 1)
        return (mean, logvar)

    def decode(self, z):
        return self.decoder(z)

    def regularization_loss(self, mean, logvar, z = None):
        if not self.use_analytical_kl: # single monte carlo sample
            return log_normal_pdf(z, mean, logvar) - log_normal_pdf(z, 0.0, 1.0)
        else: # alternatively, calculate the KL term analytically
            return 0.5 * tf.reduce_sum(tf.math.exp(logvar) + tf.math.square(mean) - 1.0 - logvar, axis = -1)

    def sample(self, samples):
        return self.decode(samples)

    def calculate_losses(self, data):
        # unpack data (input, target)
        x, y = data
        # encode
        mean, logvar = self.encode(x)
        # sample latent space
        z = self.sample_latent((mean, logvar))
        # decode
        x_hat = self.decode(z)
        # calculate reconstruction loss
        reconstruction_loss = self.compiled_loss(y, x_hat)
        # calculate regularization loss
        regularization_loss = self.regularization_loss(mean, logvar, z)
        # calculate total loss
        total_loss = reconstruction_loss + self.beta * regularization_loss
        # return losses
        return (total_loss, reconstruction_loss, regularization_loss)

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, regularization_loss = self.calculate_losses(data)
        # calculate gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        # apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)
        # return text summary
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result()
        }

    @tf.function
    def test_step(self, data):
        total_loss, reconstruction_loss, regularization_loss = self.calculate_losses(data)
        # update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)
        # return text summary
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result()
        }