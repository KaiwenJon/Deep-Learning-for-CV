import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ones = torch.ones_like(logits_real)
    zeros = torch.zeros_like(logits_fake)
    
    loss_real = bce_loss(logits_real, ones)
    loss_fake = bce_loss(logits_fake, zeros)
    loss = loss_real + loss_fake
    # we want to minimize this loss.
    # we want real to be as close to 1 as possible
    # we want fake to be as close to 0 as possible
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ones = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake, ones)
    # we want to minimize this loss
    # we want fake data to output a value as close to 1 as possible
    ##########       END      ##########
    
    return loss


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
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    loss = 0.5 * torch.mean((scores_real - 1) ** 2) + 0.5 * torch.mean(scores_fake ** 2)

    ##########       END      ##########
    
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
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    loss = 0.5 * torch.mean((scores_fake-1) ** 2)

    ##########       END      ##########
    
    return loss
