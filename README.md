# tf_py_cvae
This repository contains a generic VAE and CVAE implementation using Tensorflow 2 / Keras and Python.
Both models are implemented via the Keras subclassing API and can be freely parameterized
with encoder and decoder models.

The two generic models can be found in the files `vae.py` and `cvae.py`.
`MeanSquaredError` and `BinaryCrossentropy` builtin losses seem to
calculate the mean over the output dimension instead of the sum.
This throws off the regularization / reconstruction loss balance.
Therefore, I implemented some custom losses in `reconstruction_losses.py`.

There are also two (very messy) test files in wich I test both models
on the MNIST handwritten digit dataset.

## Disclaimer
I'm new to Tensorflow, so things might be ugly, slow or simply wrong
and may be implemented much better and more cleanly.
I just thought this might be useful as a standalone repository
for use in some of my other projects.
