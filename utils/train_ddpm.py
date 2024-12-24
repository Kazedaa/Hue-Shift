import torch
import torchvision
from torchvision.utils import make_grid

import os
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SAVE_ROOT = "outputs/DDPM"
CKPT_SAVE = "checkpoints/DDPM"

os.makedirs(IMG_SAVE_ROOT,exist_ok=True)
os.makedirs(CKPT_SAVE,exist_ok=True)

def train(
        task,
        num_epochs,
        data_loader,
        optimizer,
        T,
        scheduler,
        model,
        vae,
        criterion,
):
    for epoch in range(num_epochs):
        losses = []
        for im in tqdm(data_loader):
            optimizer.zero_grad()
            im = im.float().to(DEVICE)

            # moving from pixel space to latent space
            with torch.no_grad():
                latent_im, _, _, _ = vae.encode(im)

            if task=="l2ab":
                L,A,B = torch.split(latent_im,[1,1,1],dim=1)

            t = torch.randint(0,T,(latent_im.shape[0],)).to(DEVICE)

            noisy_im, noise = scheduler.forward(latent_im, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"{epoch} Loss {np.mean(losses)}")
        torch.save(model.state_dict(), f"{CKPT_SAVE}/ddpm.pth")