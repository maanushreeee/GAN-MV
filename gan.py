import os
import io
import zipfile
import torch
from torchvision.utils import save_image
from flask import Flask, render_template, request, jsonify, send_file, Response

# Generator definition (unchanged)
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, channels=3):
        super(Generator, self).__init__()
        self.init_size = 15
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        )
        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(512, 256, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(256, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, channels, 3, stride=1, padding=1),
            torch.nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z).view(z.size(0), 512, self.init_size, self.init_size)
        return self.conv_blocks(out)

# Global variables
LATENT_DIM = 100
MODEL_DIR = "models"  # Directory containing all model files
OUTPUT_DIR = "static/generated_images"  # Where to save generated images
MODELS = {
    "early": "Early_generator.pth",
    "benign": "Benign_generator.pth",
    "pre": "Pre_generator.pth",
    "pro": "Pro_generator.pth"
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
print(f"Using device: {device}")

def generate_images(cataract_type, image_count):
    """Generate images using the specified model and count"""
    # Clear previous generated images
    output_subdir = os.path.join(OUTPUT_DIR, cataract_type)
    if os.path.exists(output_subdir):
        for file in os.listdir(output_subdir):
            file_path = os.path.join(output_subdir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    else:
        os.makedirs(output_subdir)
    
    # Load the appropriate model
    model_path = os.path.join(MODEL_DIR, MODELS[cataract_type])
    generator = Generator(LATENT_DIM).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Generate images
    image_paths = []
    with torch.no_grad():
        for i in range(int(image_count)):
            z = torch.randn(1, LATENT_DIM, device=device)
            gen_img = generator(z)
            gen_img = (gen_img + 1) / 2  # Denormalize to [0,1]
            
            # Save the image
            img_filename = f"{cataract_type}_gen_{i+1:03d}.png"
            img_path = os.path.join(output_subdir, img_filename)
            save_image(gen_img, img_path)
            
            # Store relative path for front-end
            # In your generate_images function, modify this line:
            image_paths.append(os.path.join("static", "generated_images", cataract_type, img_filename))
    
    return image_paths

def create_zip_file(cataract_type):
    """Create a zip file containing all generated images"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for file in os.listdir(os.path.join(OUTPUT_DIR, cataract_type)):
            file_path = os.path.join(OUTPUT_DIR, cataract_type, file)
            if os.path.isfile(file_path):
                zf.write(file_path, arcname=file)
    
    memory_file.seek(0)
    return memory_file