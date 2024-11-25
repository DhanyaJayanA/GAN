#### Generative AI 

Create synthetic contents (image, text audio and video).

#### Generative Adversarial Network (GAN)

Ian J. Goodfellow et al., 2014 proposed the first generative model is generative adversarial network (GAN), which generates synthetic images.

#### Deep Convolutional Generative Adversarial Network (DCGAN)

DCGAN consist of discriminator and generator model. The generator produces fake images which look like training images. The generator tries to find the best image to fool the discriminator. The discriminator tries to classify if the image is real or fake.
   
         Architecture guidelines for stable Deep Convolutional GANs

         • Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).

         • Use batchnorm in both the generator and the discriminator.

         • Remove fully connected hidden layers for deeper architectures.

         • Use ReLU activation in generator for all layers except for the output, which uses Tanh.

         • Use LeakyReLU activation in the discriminator for all layers.

**DCGAN Architecture (128x128X3):**     
<img src="https://github.com/DhanyaJayanA/Generative-AI/blob/main/Untitled.jpg" alt="DCGAN">        
                 
**Problem Statement: Generate Fake Image using DCGAN**

In this project discriminator consists of strided Conv2D layers and LeakyRelu as activation function. The generator consists of Conv2DTranspose layers, batch normalization layers, **LeakyRelu activation** is used for all the layers except the last layer which uses **tanh/sigmoid**. The output will be a 3x64x64 RGB fake image.

<a href="https://github.com/DhanyaJayanA/Generative-AI/blob/main/GenerateFakeImage_DCGAN.ipynb">Code</a>

#### Variational autoencoder (VAE)

Diederik P Kingma & Max, 2013 proposed autoencoder variational bayes (VAE) an extension of the traditional autoencoder.  The encoder map the input data into the latent space and the decoder reconstruct it from latent space.

#### Diffusion models

Jascha Sohl-Dickstein et al., 2015 proposed the first diffusion probabilistic model (DPM) using non-equilibrium thermodynamics. In the forward process add noise and in the reverse process restore it. 

