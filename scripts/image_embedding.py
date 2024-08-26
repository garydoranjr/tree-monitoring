#Import necessary libraries/modules
"""
Author: Luis Hernandez
"""
import os
import click
import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
from tqdm import tqdm
from glob import glob


def load_image(img_path: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img_path).convert('RGB')

    # Define the transformation
    transform_image = T.Compose([
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5])
    ])

    transformed_img = transform_image(img).unsqueeze(0)
    return transformed_img

def compute_embeddings(model, device, input_dir, output_file) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """

    files = glob(os.path.join(input_dir, '*.tif'))

    all_embeddings = {}
    
    with torch.no_grad():
        for img_path in tqdm(files, desc="Processing files"):
            image_id = os.path.splitext(os.path.basename(img_path))[0]
            img_tensor = load_image(img_path).to(device)
            embeddings = model(img_tensor)
            all_embeddings[image_id] = embeddings[0].cpu().numpy().tolist()

    with open(output_file, "w") as f:
        json.dump(all_embeddings, f, indent=2)

    return all_embeddings


@click.command()
@click.argument('crown_folder')
@click.argument('output_folder')
def main(crown_folder, output_folder):

    subdirs = glob(os.path.join(crown_folder, '*'))

    # Load the model
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    for subdir in tqdm(subdirs, 'Extracting Features'):
        tag_id = os.path.basename(subdir)
        output_file = os.path.join(output_folder, f'{tag_id}.json')
        if os.path.exists(output_file): continue

        # Compute embeddings
        embeddings = compute_embeddings(dinov2_vits14, device, subdir, output_file)


if __name__ == '__main__':
    main()
