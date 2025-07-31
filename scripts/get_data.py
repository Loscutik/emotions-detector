from pathlib import Path
import yaml

import numpy as np

data_dir = Path("../data")

def fetch_file(url: str, folder: str|Path, filename: str | None = None) -> str:
    """
    get a file from the given url  and save it to the 'folder' with the given filename.
    Creates the folder if it doesn't exist
    """
    import requests
    if filename is None:
        filename = url.split('/')[-1]
    if isinstance(folder, str):
        folder = Path(folder)
    path_destination = folder / filename
    if not folder.exists():
        folder.mkdir(parents=True)
    if not path_destination.exists():
        response = requests.get(url,stream=True)
        with open(path_destination, 'wb') as file_destination:
            for chunk in response.iter_content(chunk_size=128):
                file_destination.write(chunk)

    return path_destination

def unzip_file(path_zip: str, dir_destination: str) -> None:
    """
    extract files from `path_zip` to destination `dir_destination` directory
    """
    import zipfile

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(dir_destination)
        

def read_configuration(config_path:str|Path)->tuple[dict, dict, list]:
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    for key, value in data['paths'].items():
        data['paths'][key] = Path(value)
        if '_dir' in key and not data['paths'][key].exists():
            raise FileNotFoundError(f"Path {data['paths'][key]} does not exist. Please check the config file.")
    return data

img_size=48  # or whatever size your images should be
def normalize_images(imgs:np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to the range [0, 1].

    Args:
        imgs (np.ndarray): Array of images with pixel values in the range [0, 255].

    Returns:
        np.ndarray: Array of images with pixel values normalized to [0, 1].
    """
    return imgs/255.

def str2pixels(s:str, image_w:int=img_size) -> np.ndarray:
    """
    Convert a string of space-separated pixel values into a 2D NumPy array.

    Args:
        s (str): String containing space-separated pixel values.
        image_w (int, optional): Width (and height) of the output image array. Defaults to img_size.

    Returns:
        np.ndarray: 2D array of shape (image_w, image_w) with dtype uint8.
    """
    return np.array(s.split(' '), dtype=np.uint8).reshape(-1, image_w)

def read_data(path:str, img_size:int=img_size, out_dtype=np.float32) -> tuple[np.ndarray, np.ndarray]   :
    """
    Read image data and labels from a CSV file, process the images, and return them as NumPy arrays.

    The function reads a CSV file where each row contains an emotion label and a string of pixel values.
    It converts the pixel strings to 2D arrays, applies histogram equalization for better contrast,
    normalizes the images to [0, 1], and adds a channel dimension. The labels are returned as a NumPy array.

    Args:
        path (str): Path to the CSV file containing the data.
        img_size (int, optional): Size (width and height) of the images. Defaults to img_size.
        out_dtype (data-type, optional): Desired output data type for the images. Defaults to np.float32.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - X (np.ndarray): Array of processed images with shape (num_samples, img_size, img_size, 1).
            - y (np.ndarray): Array of emotion labels.
    """
    from pandas import read_csv
    from cv2 import equalizeHist
    df = read_csv(path,dtype={'emotion':np.uint8},  converters={'pixels': lambda x: str2pixels(x, image_w=img_size)}, usecols=['emotion', 'pixels'])
    Xx = np.empty((len(df), img_size, img_size), dtype=np.uint8)
    i = 0
    for arr in df['pixels']:
        Xx[i,:,:] = arr
        Xx[i,:,:] = equalizeHist(arr) # equalize histogram for better contrast
        i+=1
    return np.expand_dims(normalize_images(Xx), axis=-1).astype(out_dtype), df['emotion'].to_numpy()  #normalize Xx and add to it a dimension for channels 
