import torch
import torch.nn as nn
import torchvision
import numpy as np
from models import *

latent_dim = 100
data_dim = 28 * 28
device = "cuda"
batch_size = 16
num_epochs = 100

dataset = torchvision.datasets.MNIST(
    "MNIST",
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = Generator(latent_dim=latent_dim, data_dim=data_dim).to(device)
D = Discriminator(data_dim=data_dim).to(device)

optimizer_G = torch.optim.SGD(G.parameters(), lr=1e-3)
optimizer_D = torch.optim.SGD(D.parameters(), lr=1e-3)

criterion = nn.BCELoss()

Z = torch.clone(torch.nn.utils.parameters_to_vector(D.parameters()).detach())
p = 10
beta = 0.5

loss_Ds = list()
loss_Gs = list()
loss_Zs = list()

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (img_batch, _) in enumerate(dataloader, 0):
        img_batch = torch.flatten(img_batch.to(device), start_dim=1)
        bs = img_batch.shape[0]

        ######## Discriminator ########
        D.zero_grad()

        # Real data
        y = torch.ones(size=(bs, 1), device=device, dtype=torch.float)
        real_output_D = D(img_batch)

        a = torch.sum(torch.isnan(real_output_D))
        assert a == 0

        real_loss_D = criterion(real_output_D, y)
        real_loss_D.backward()

        # Fake data
        r = torch.randn(bs, latent_dim, device=device)
        y = torch.zeros(size=(bs, 1), device=device, dtype=torch.float)
        fake_images = G(r)
        fake_output_D = D(fake_images.detach())

        a = torch.sum(torch.isnan(fake_output_D))
        assert a == 0

        fake_loss_D = criterion(fake_output_D, y)
        fake_loss_D.backward()

        loss_Z = (
            p
            / 2
            * torch.sum(
                torch.square(torch.nn.utils.parameters_to_vector(D.parameters()) - Z)
            )
        )
        loss_Z.backward()

        loss_D = real_loss_D + fake_loss_D

        loss_Ds.append(float(loss_D))
        loss_Zs.append(float(loss_Z))

        optimizer_D.step()

        ######## Generator ########
        G.zero_grad()

        y = torch.ones(size=(bs, 1), device=device, dtype=torch.float)
        output_G = D(fake_images)

        a = torch.sum(torch.isnan(output_G))
        assert a == 0

        loss_G = criterion(output_G, y)
        loss_G.backward()

        loss_Gs.append(float(loss_G))

        optimizer_G.step()

        params_D = torch.nn.utils.parameters_to_vector(D.parameters()).detach()
        Z = Z + beta * (params_D - Z)

        if i % 500 == 0:
            print(
                f"[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {loss_D.item():.4f}\tLoss_G: {loss_G.item():.4f}\tLoss_Z: {float(loss_Z):.10f}"
            )
