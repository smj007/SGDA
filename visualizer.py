import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import *

# Save while training and load

latent_dim = 100
data_dim = 28 * 28
device = "cpu"
loss_Ds = list()
loss_Gs = list()
loss_Zs = list()
G = Generator(latent_dim=latent_dim, data_dim=data_dim).to(
    device
)  # Save weights while training and load


def vis(latent_dim=100, device="cpu"):
    fixed_noise = torch.randn(64, latent_dim, device=device)
    img_list = list()
    with torch.no_grad():
        fake = G(fixed_noise).detach().cpu().reshape((64, 28, 28))

    plt.figure(figsize=(10, 10))
    N = 4
    for i in range(N**2):
        plt.subplot(N, N, i + 1)
        plt.imshow(fake[i + 10], cmap="gray")
        plt.axis("off")
    plt.show()

    plt.plot(loss_Ds, label="Discriminator")
    plt.plot(loss_Gs, label="Generator")
    plt.plot(loss_Zs, label="Z")
    plt.legend()
    plt.show()

    plt.plot(loss_Ds, label="Discriminator")
    plt.plot(loss_Gs, label="Generator")
    plt.plot(loss_Zs, label="Z")
    plt.yscale("log")
    plt.legend()
    plt.show()
