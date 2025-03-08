# DCGAN on CelebA Dataset

This repository contains an implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the CelebA dataset. The model is designed to generate realistic face images after training.

## Overview

The project leverages a DCGAN architecture with a Generator and a Discriminator network. The Generator learns to produce fake images that mimic real CelebA images, while the Discriminator learns to distinguish between real and generated images. Training involves an adversarial process where both networks improve iteratively.

## DCGAN Architecture
![DCGAN Architecture](./DCGAN%20Architecture.png)

## Dataset Preprocessing Steps

1. **Dataset Download**:  
   The CelebA dataset is automatically downloaded using the torchvision dataset utility. The dataset is expected to be stored in the `./data/celeba` directory.

2. **Transformations**:  
   The preprocessing includes:
   - **Center Crop**: The original images are center-cropped to focus on the face region.
   - **Resize**: The images are resized to `64x64` pixels.
   - **Normalization**: The pixel values are normalized to the range \([-1, 1]\) which is optimal for the DCGAN model using a Tanh activation function in the Generator’s output layer.

3. **DataLoader Setup**:  
   A DataLoader is created with a specified batch size (128) and the number of workers (2) to efficiently load and batch the preprocessed images during training.

## How to Train and Test the Model

### Training

1. **Setup Environment**:  
   Ensure that PyTorch and torchvision are installed. The script automatically detects GPU availability (`cuda:0`) and falls back to CPU if necessary.

2. **Initialization**:  
   - Set random seeds for reproducibility.
   - Define hyperparameters including latent vector size (`nz`), generator/discriminator feature map sizes (`ngf`, `ndf`), learning rate, and number of training epochs.
   - Initialize the weights of both the Generator and Discriminator networks.

3. **Training Loop**:  
   The training loop consists of the following steps for each batch:
   - **Discriminator Update**:
     - Process a batch of real images from the dataset.
     - Generate fake images using the Generator.
     - Compute the loss for both real and fake images using Binary Cross Entropy Loss (BCE Loss).
     - Backpropagate the loss and update the Discriminator’s parameters.
   - **Generator Update**:
     - Generate a new batch of fake images.
     - Compute the loss based on the Discriminator’s feedback.
     - Backpropagate and update the Generator’s parameters.
   - **Logging and Saving**:
     - Training statistics (losses and discriminator outputs) are printed every 50 batches.
     - Generated images are saved to the `results` directory after each epoch.
     - The progress of both networks is tracked for later visualization.

4. **Monitoring Training**:  
   - The loss curves for both the Generator and Discriminator are plotted to monitor convergence.
   - A fixed noise vector is used to generate images after each epoch, ensuring that improvements in image quality can be visually tracked over time.

### Testing

1. **Generate Images**:  
   Once training is complete, the Generator is used in inference mode:
   - A fixed noise vector is fed into the Generator to produce output images.
   - The generated images are saved and visualized using the provided `imshow` function.

2. **Visual Evaluation**:  
   - The saved images and plotted loss curves help in qualitatively and quantitatively assessing the model's performance.
   - Expected outputs include a grid of face-like images and a plot that shows the Generator and Discriminator loss over training iterations.

## Expected Outputs

- **Training Logs**:  
  During training, logs display the current epoch, batch progress, Discriminator loss, Generator loss, and the probability outputs for both real and fake images.
  
- **Generated Images**:  
  After each epoch, generated images (saved in the `results` directory) should resemble faces as the Generator learns to mimic the CelebA dataset.
  
- **Loss Curves**:  
  A plot showing the loss trajectories of both the Generator and Discriminator, which provides insight into the training dynamics.

- **Final Output Visualization**:  
  The final generated image grid is displayed using the `imshow` function, allowing you to visually inspect the quality of the generated samples.

---

This README provides the necessary instructions and information to run the DCGAN training and testing pipeline on the CelebA dataset.
