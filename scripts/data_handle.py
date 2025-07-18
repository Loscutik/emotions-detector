from typing import Iterable, Callable
from pathlib import Path
import yaml

import numpy as np
import tensorflow as tf

from pydantic import BaseModel

import cv2

data_dir = Path("../data")

def read_configuration(config_path:str|Path)->tuple[dict, dict, list]:
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
        CONFIG = data['config']
        PATHS = data['paths']
        EMOTIONS = data['emotions']
    for key, value in PATHS.items():
        if '_dir' in key:
            PATHS[key] = Path(value)
            if not PATHS[key].exists():
                raise FileNotFoundError(f"Path {PATHS[key]} does not exist. Please check the config file.")
    return CONFIG, PATHS, EMOTIONS

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


def get_io_indices(interpreter:tf.lite.Interpreter) -> tuple[int, int]:
    """
    Retrieve the input and output tensor indices from a TensorFlow Lite interpreter.

    Args:
        interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter instance.

    Returns:
        tuple[int, int]: A tuple containing the input tensor index and output tensor index.
    """
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    return input_index, output_index


def img_prepare(img:np.ndarray, img_shape=(img_size, img_size), flt_type=np.float32)-> np.ndarray:
    """
    Prepare an image for model prediction by normalizing, resizing, and adding necessary dimensions.

    Args:
        img (np.ndarray): Input image array.
        img_shape (tuple, optional): Desired shape (width, height) for resizing the image. Defaults to (img_size, img_size).
        flt_type (data-type, optional): Data type to cast the image to before normalization. Defaults to np.float32.

    Returns:
        np.ndarray: Prepared image array with shape (1, img_shape[0], img_shape[1], 1).
    """
    img = normalize_images(np.astype(img, flt_type))
    img = cv2.resize(img, img_shape)[np.newaxis,:,:,np.newaxis]
    return img


def predict_img_tflite(img:np.ndarray, interpreter:tf.lite.Interpreter, 
                       input_index:int|None=None, output_index:int|None=None, 
                       img_shape:tuple[int,int]|None=None, flt_type=None,
                       img_prepare_fn:Callable=img_prepare) -> np.ndarray:
    
    if input_index is None or output_index is None or img_shape is None or flt_type is None:
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_index or input_details['index']
        output_index = output_index or output_details['index']
        img_shape = img_shape or tuple(input_details['shape'][1:3])
        flt_type = flt_type or input_details['dtype']

    interpreter.allocate_tensors()

    img_prepared = img_prepare_fn(img, img_shape=img_shape, flt_type=flt_type)

    # # Квантизація (для uint8/int8 моделей)
    # input_dtype = input_details[0]['dtype']
    # if input_dtype == np.uint8 or input_dtype == np.int8:
    #     scale, zero_point = input_details[0]['quantization']
    #     img_prepared = (img_prepared / scale + zero_point).numpy().astype(input_dtype)
    # else:
    #     img_prepared = tf.cast(img_prepared, input_dtype).numpy()

    # Prediction
    interpreter.set_tensor(input_index, img_prepared)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)

    return output[0] # output is a 2 demention array, length = 1


def predict_batch_tflite(x:Iterable, interpreter:tuple[tf.lite.Interpreter, int,int], 
                         input_index:int|None=None, output_index:int|None=None, 
                         input_shape:Iterable|None=None, output_shape:Iterable|None=None, output_type=None,
                         batch_size:int=32)-> np.ndarray:

    if input_index is None or output_index is None or input_shape is None or output_type is None:
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_index = input_index or input_details['index']
        output_index = output_index or output_details['index']
        input_shape = input_shape or input_details['shape']
        output_shape = output_shape or output_details['shape']
        output_type = output_type or input_details['dtype']

    interpreter.resize_tensor_input(input_index, (batch_size, *input_shape[1:]))
    interpreter.allocate_tensors()

    # Prediction
    total_samples = len(x)
    total_batches = (total_samples + batch_size - 1) // batch_size
    output = np.zeros((total_samples, *output_shape[1:]), dtype=output_type)
    for i in range(total_batches):
        print('.', end='')
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        if batch_end > total_samples:
            batch_end = total_samples
            interpreter.resize_tensor_input(input_index, (batch_end-batch_start, *input_shape[1:]))
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, x[batch_start:batch_end])
        interpreter.invoke()
        output[batch_start:batch_end] = interpreter.get_tensor(output_index)

    print(end='\n')
    return output

def predict_img_tf(img:np.ndarray, model:tf.keras.Model, 
                   img_shape:tuple[int,int]|None=(img_size, img_size), flt_type=np.float32,
                   img_prepare_fn:Callable=img_prepare)-> np.ndarray:
    if img_shape is None or flt_type is None:
        img_shape = model.input_shape[1:3]
        flt_type = model.input_dtype
        
    img_prepared = img_prepare_fn(img, img_shape=img_shape, flt_type=flt_type)
    output = model(img_prepared, training=False).numpy()[0] #model.predict is designed for batch processing
    return output

def predict_batch_tf(x:Iterable, model:tf.keras.Model)-> np.ndarray:
    output = model.predict(x, verbose=1) # predict is desined for batch processing
    return output

def evaluate_tflite_model(interpreter:tf.lite.Interpreter, 
                          x:Iterable, y:Iterable, 
                          metrics:str|Callable|Iterable[str|Callable]|None=None, loss:str|Callable|None="CategoricalCrossentropy", 
                          return_dict=False)-> dict|list:
    class Metric(BaseModel):
        name: str
        fn: Callable
        val: float
    # Get metrics functions
    if metrics is None:
        metrics = ["accuracy"]
    elif isinstance(metrics, str):
        metrics = [metrics]

    if "accuracy" in metrics:
        idx = metrics.index("accuracy")
        if y.shape[-1] == 1: 
            metrics[idx] = "binary_accuracy"
        else:
            metrics[idx] = "categorical_accuracy"

    # Get loss functions
    loss_fn = None
    if isinstance(loss, str):
        try:
            loss_fn = tf.keras.losses.get(loss)
        except ValueError:
            raise ValueError(f"Can't recognize the function: '{loss}'")
    elif callable(loss):
        loss_fn = loss

    # Init metrics 
    def get_metric_fn(metric_name):
        if isinstance(metric_name, str):
            try:
                return tf.keras.metrics.get(metric_name)
            except ValueError as e:
                raise ValueError(f"Can't recognize the metric function: {metric_name}") from e   
        elif callable(metric_name): 
            return metric_name
        else:
            raise ValueError(f"Metric should be a string or callable, got {type(metric_name)}")
        
    metric_objs = [Metric(name=metric, fn=get_metric_fn(metric), val=0.0) for metric in metrics]

    if loss_fn is not None:
        metric_objs.append(Metric(name='loss', fn=loss_fn, val=0.0))

    # Evaluate
    output = predict_batch_tflite(x, interpreter)

    for metric_o in metric_objs:
        metric_o.val += metric_o.fn(y, output).numpy()


    results = {mo.name: mo.val for mo in metric_objs}
    print(results)

    return results if return_dict else list(results.values())