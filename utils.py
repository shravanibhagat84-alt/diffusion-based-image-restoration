import torch
from torchvision import transforms
from PIL import Image

def load_image(path, size=(256,256)):
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    tensor = transform(img)  # This returns CxHxW (3D) for RGB images
    
    # Ensure exactly 3D (C, H, W) - remove any extra dimensions
    while tensor.dim() > 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        else:
            break
    
    return tensor  # Should be CxHxW

def save_image(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor.clamp(0,1), path)
