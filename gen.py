import torch
from model import Generator  
import config
from torchvision.utils import save_image
from scipy.stats import truncnorm


def generate_images(model, steps, truncation=0.7, n=10):
    model.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(
                truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)),
                device=config.DEVICE,
                dtype=torch.float32,
            )
            img = model(noise, alpha, steps)
            save_image(img * 0.5 + 0.5, f"static/saved_images/img_{i}.png")
    model.train()

if __name__ == "__main__":
    Z_DIM = 256
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3).to(config.DEVICE)

    checkpoint_file = "generator.pth"  
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    
    steps = 5  

    generate_images(gen, steps)
