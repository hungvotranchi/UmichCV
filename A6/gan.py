from __future__ import print_function

import torch
import torch.utils.data
from torch import conv2d, nn, optim

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise = torch.FloatTensor(batch_size, noise_dim).uniform_(-1, 1).to(device)

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Linear(784, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, 256),
      nn.LeakyReLU(0.01),
      nn.Linear(256, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Linear(noise_dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 784),
      nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    true_real = torch.ones_like(logits_real)
    loss_real = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, true_real)

    true_fake = torch.zeros_like(logits_fake)
    loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, true_fake)
    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    true_fake = torch.ones_like(logits_fake)
    loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, true_fake)
    loss = loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,betas=(0.5,0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    true_real = torch.ones_like(scores_real)
    loss_real = torch.nn.functional.mse_loss(scores_real, true_real)
    
    # For fake samples: minimize (D(G(z)))^2
    true_fake = torch.zeros_like(scores_fake)
    loss_fake = torch.nn.functional.mse_loss(scores_fake, true_fake)
    
    # Total loss
    loss = 0.5 * (loss_real + loss_fake)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    true_fake = torch.ones_like(scores_fake)
    loss_fake = torch.nn.functional.mse_loss(true_fake, scores_fake)
    loss = 0.5 * loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Unflatten(dim=1,unflattened_size= (1, 28, 28)),
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1),
      nn.LeakyReLU(0.01),
      nn.MaxPool2d(kernel_size=2,stride=2),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=1),
      nn.LeakyReLU(0.01),
      nn.MaxPool2d(kernel_size=2,stride=2),
      nn.Flatten(),
      nn.Linear(1024, 1024),
      nn.LeakyReLU(0.01),
      nn.Linear(1024, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
      nn.Linear(noise_dim, 1024),
      nn.ReLU(),
      nn.BatchNorm1d(num_features=1024),
      nn.Linear(1024, 7*7*128),
      nn.ReLU(),
      nn.BatchNorm1d(num_features=7*7*128),
      nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
      nn.ConvTranspose2d(in_channels=128,out_channels=64, kernel_size=(4,4), stride=2, padding = 1),
      nn.ReLU(),
      nn.BatchNorm2d(num_features=64),
      nn.ConvTranspose2d(in_channels=64, out_channels=1,kernel_size=(4,4), stride=2, padding=1),
      nn.Tanh(),
      nn.Flatten()

    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
