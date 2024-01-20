from diffusers import AutoencoderKL
import torch

def build_vae_corruption(opt, log, vae_model_name):
    class VAECorruption:
        def __init__(self, vae_model_name, device):
            self.device = device
            self.vae = AutoencoderKL.from_pretrained(vae_model_name).to(self.device)

        def __call__(self, images):
            with torch.no_grad():
                # Ensure images are on the correct device
                images = images.to(self.device)

                # Encode and Decode using VAE
                encoded_samples = self.vae.encode(images).latent_dist.sample()
                decoded_samples = self.vae.decode(encoded_samples).sample
                
            return decoded_samples

    return VAECorruption(vae_model_name, opt.device)
