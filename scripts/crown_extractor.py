#Import libraries/modules
import os
import click
import geopandas as gpd
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm

#Change tag ID to whatever tree species you would like to extract
tag_id = 'your_tag_id'

#Load images from dataset folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith("rgb.tif"):
            images.append(os.path.join(folder, filename))
    return images

#Load the shapefile
def load_shapefile(shapefile_path):
    return gpd.read_file(shapefile_path)

#Get the tree crown based on provided tag ID
def get_tree_crown(geodataframe, tag_id):
    tree_crowns = geodataframe[geodataframe['tag'] == tag_id]
    if tree_crowns.empty:
        return None
    return tree_crowns.geometry.values[0]

#Extract a 25x25 window around the tree crown
def extract_window(image_path, tree_crown, window_size=25):
    with rasterio.open(image_path) as src:
        bounds = tree_crown.bounds
        center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

        # Calculate pixel coordinates
        center_row, center_col = src.index(center_x, center_y)
        half_window = window_size // 2

        row_start = max(center_row - half_window, 0)
        row_end = min(center_row + half_window, src.height)
        col_start = max(center_col - half_window, 0)
        col_end = min(center_col + half_window, src.width)

        if (
            (col_end - col_start < (window_size - 1)) or
            (row_end - row_start < (window_size - 1))
        ):
            raise ValueError('Outside bounds')

        # Adjust window size if it exceeds image boundaries
        if row_end - row_start < window_size:
            row_start = max(0, row_end - window_size)
            row_end = row_start + window_size
        if col_end - col_start < window_size:
            col_start = max(0, col_end - window_size)
            col_end = col_start + window_size

        # Ensure window is within image boundaries
        if row_end > src.height:
            row_end = src.height
            row_start = row_end - window_size
        if col_end > src.width:
            col_end = src.width
            col_start = col_end - window_size

        window = src.read(window=((row_start, row_end), (col_start, col_end)))
        window_transform = src.window_transform(((row_start, row_end), (col_start, col_end)))
        return window, window_transform

#Save the extracted window
def save_window(window, output_folder, filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename)
    
    # Convert the numpy array to an Image
    image = Image.fromarray(np.transpose(window, axes=(1, 2, 0)).astype(np.uint8))
    image.save(output_path)
    return output_path

#Process the images
def process_images(image_folder, shapefile_path, tag_id, output_folder):
    images = load_images_from_folder(image_folder)
    
    shapefile = load_shapefile(shapefile_path)
    
    tree_crown = get_tree_crown(shapefile, tag_id)
    if tree_crown is None:
        print(f"Tree crown with tag ID {tag_id} not found in shapefile.")
        return
    
    for image_path in images:
        try:
            window, _ = extract_window(image_path, tree_crown)
        except ValueError:
            continue
        filename = os.path.basename(image_path)
        save_path = save_window(window, output_folder, filename)


@click.command()
@click.argument('image_folder')
@click.argument('shapefile_path')
@click.argument('output_folder')
def main(image_folder, shapefile_path, output_folder):

    shp = load_shapefile(shapefile_path)
    tags = np.unique(shp['tag'])
    tags = sorted([t for t in tags if int(t) > 0])

    for tag_id in tqdm(tags[::50], 'Extracting Crowns'):
        outdir = os.path.join(output_folder, tag_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            continue
        process_images(image_folder, shapefile_path, tag_id, outdir)

## Specify the paths
#image_folder = 'your/file/path'
#shapefile_path = 'your/file/path'
#output_folder = 'your/file/path'+tag_id+'folder'
#output_folder_outline = 'your/file/path'+tag_id+'folder'

if __name__ == '__main__':
    main()
