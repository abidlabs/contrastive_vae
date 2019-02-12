# Contrastive VAE

Python code for learning salient latent features with contrastive variational autoencoders. This code is useful when one is interested in patterns or latent variables that exist one dataset, but not the other. Applications include dicovering subgroups in biological and medical data. 

See the original paper, titled "Contrastive Variational Autoencoder Enhances Salient Features" by Abubakar Abid and James Zou, which is available on ArXiv and submitted to ICML 2019.

Key code files are:

* `utils.py`, which contains general utility files
* `vae_utils.py`, which contains files that define and train a fully-connected VAE and cVAE
* `celeb_utils.py`, which contains files that define and train a convolutional VAE and cVAE

We also include jupyter notebooks that demonstrate how to use these algorithms in a variety of different settings

