## Denoising Autoencoder

### Introduction

Autoencoder are type of nerual network which are trained to replicate the orignal data. It is a unsupervised learning technique. It is used for removing noise, dimensionality reduction, feature extraction, and learning generative models of data. They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation. This kind of network is composed of two parts: an encoder and a decoder. The encoder compresses the input and produces the latent-space representation, and the decoder then reconstructs the original input only using this representation.

### Training results

The model was trained for 10 epochs on MNIST dataset. The loss function decreased with each epoch. The loss function for the last epoch was 0.9753. After 10 epochs model generated the following output on test data.

<img src='training-results.png' />
