import torch
import torchvision
from torchvision.utils import make_grid

import os
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SAVE_ROOT = "outputs/VAE"
CKPT_SAVE = "checkpoints/VAE"


os.makedirs(IMG_SAVE_ROOT,exist_ok=True)
os.makedirs(CKPT_SAVE,exist_ok=True)

def train(
        model,
        discriminator,
        lpips_model,
        num_epochs,
        data_loader,
        optimizer_g,
        optimizer_d,
        recon_criterion,
        adv_criterion,
        adv_start,
        sample_step

):
    step_count = 0
    sample_no = 0
    for epoch in range(num_epochs):
        recon_losses = []
        kl_losses = []
        lpips_losses = []
        g_losses = []
        d_losses = []
        for im in tqdm(data_loader):
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            step_count+=1

            im = im.float().to(DEVICE)

            # Generator
            model_output = model(im)
            output, encoder_out, mean, logvar = model_output    

            recon_loss = recon_criterion(output, im)
            kl_loss = torch.mean( 0.5 * torch.sum( torch.exp(logvar) + mean**2 - logvar - 1, dim=1) )

            g_loss = recon_loss + 0.00001 * kl_loss 

            if step_count > adv_start:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = adv_criterion(disc_fake_pred,
                                               torch.ones(disc_fake_pred.shape,
                                                          device = disc_fake_pred.devuce()))
                g_loss += 0.5 * disc_fake_loss

            lpips_loss = torch.mean(lpips_model(output, im))
            g_loss += lpips_loss
            g_loss.backward(retain_graph = True)
            optimizer_g.step()

            # Save sample
            if step_count % sample_step == 0 or step_count==1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                img.save(f"{IMG_SAVE_ROOT}/{sample_no}.jpg")
                sample_no += 1

            # Discriminator
            if step_count > adv_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = adv_criterion(disc_fake_pred,
                                               torch.ones_like(disc_fake_pred,
                                                          device = disc_fake_pred.device()))
                disc_real_loss = adv_criterion(disc_real_pred,
                                               torch.zeros_like(disc_real_pred,
                                                          device = disc_real_pred.device()))   
                d_loss = 0.5 * ( disc_real_loss + disc_fake_loss)    
                d_loss.backward()         
                optimizer_d.step()
                d_losses.append(d_loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            lpips_losses.append(lpips_loss.item())
            g_losses.append(g_loss.item())
        print(f"Epoch {epoch} recon_loss {np.mean(recon_losses)} kl_loss {np.mean(kl_losses)} lpips_loss {np.mean(lpips_losses)} g_loss {np.mean(g_losses)} d_loss {np.mean(d_losses)}")
        torch.save(model.state_dict(), f"{CKPT_SAVE}/model.pth")
        torch.save(discriminator.state_dict(), f"{CKPT_SAVE}/discriminator.pth")