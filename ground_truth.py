import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def generate_color_trimaps(trimap_dir, output_dir, image_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    trimap_files = sorted([
        f for f in os.listdir(trimap_dir)
        if f.lower().endswith(".png") and not f.startswith("._")
    ])

    for trimap_file in trimap_files:
        trimap_path = os.path.join(trimap_dir, trimap_file)
        trimap = np.array(Image.open(trimap_path).convert("L"))

        # Resize to target size if needed
        trimap = np.array(Image.fromarray(trimap).resize(image_size, resample=Image.NEAREST))

        # Normalize to [0, 1] for colormap application
        normalized = (trimap.astype(np.float32) - trimap.min()) / (trimap.max() - trimap.min())

        # Apply jet colormap
        colormap = cm.get_cmap('jet')
        colored = colormap(normalized)[:, :, :3]  # remove alpha channel
        colored_img = (colored * 255).astype(np.uint8)

        # Save colorized image
        save_path = os.path.join(output_dir, trimap_file)
        Image.fromarray(colored_img).save(save_path)

