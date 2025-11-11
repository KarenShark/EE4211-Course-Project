import os
import urllib.request
import tarfile
from ground_truth import generate_color_trimaps

def download_data(data_path='./data'):

    """
    Download and extract the Oxford-IIIT Pet dataset if it doesn't already exist in the specified path.
    Also generates ground truth masks from trimaps and saves them to the 'ground-truth' folder if they don't already exist.
    Args:
        data_path (str): Path where the dataset should be saved. Defaults to './data'.
    """

    urls = {
        "images.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "annotations.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    }

    if not os.path.exists(data_path):
        print(f"'{data_path}' not found. Downloading dataset...")
        os.makedirs(data_path, exist_ok=True)
        for filename, url in urls.items():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Extracting {filename} to {data_path}...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=data_path)
            os.remove(filename)
            print(f"{filename} processed and removed.")
        print("Data download and extraction complete.\n")
    else:
        print(f"'{data_path}' already exists — skipping dataset download.")

    

    #generate trimap ground truths if it doesn't exist
    gt_dir = "ground-truth"
    if not os.path.exists(gt_dir):
        print("Generating ground-truth masks from trimaps...")
        generate_color_trimaps(
        trimap_dir="data/annotations/trimaps",
        output_dir="ground-truth",
        image_size=(224, 224)  
    )
    else:
        print(f"'{gt_dir}' folder already exists — skipping ground truth image generation.")