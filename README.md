
#### Deep Convolutional Generative Adversarial Network (DCGAN)

DCGAN consist of discriminator and generator model. The generator produces fake images which look like training images. The generator tries to find the best image to fool the discriminator. The discriminator tries to classify if the image is real or fake.
   
         Architecture guidelines for stable Deep Convolutional GANs

         • Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).

         • Use batchnorm in both the generator and the discriminator.

         • Remove fully connected hidden layers for deeper architectures.

         • Use ReLU activation in generator for all layers except for the output, which uses Tanh.

         • Use LeakyReLU activation in the discriminator for all layers.
         
                 
**Problem Statement: Generate Fake Image using DCGAN**

In this project discriminator consists of strided Conv2D layers and LeakyRelu as activation function. The generator consists of Conv2DTranspose layers, batch normalization layers, **LeakyRelu activation** is used for all the layers except the last layer which uses **tanh/sigmoid**. The output will be a 3x64x64 RGB fake image.

<a href="https://github.com/DhanyaJayanA/DCGAN/blob/main/GenerateFakeImage_DCGAN.ipynb">Code</a>
