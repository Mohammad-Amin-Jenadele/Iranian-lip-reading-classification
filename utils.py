# Package import
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import imageio
from io import BytesIO
import base64
from IPython.display import HTML
import os
import cv2
from typing import List, Tuple, Optional , Generator


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Precision score.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Recall score.
    """
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def plot_confusion_matrix(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> None:
    """
    Plot confusion matrix.

    Parameters:
    - model: Trained model.
    - x: Input data.
    - y: True labels.
    """
    y_pred = model.predict(x)
    y_pred_plot = np.argmax(y_pred, axis=1)
    y_test_plot = np.argmax(y, axis=1)

    cm = confusion_matrix(y_test_plot, y_pred_plot)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


def show_MP4(files: list[str], labels: list[int], num_to_class_dict: dict[int : str], number_of_samples: int) -> HTML:
    """
    Display a random selection of MP4 files as GIFs with their corresponding labels.

    Args:
    - files (List[str]): A list of strings representing file paths to MP4 files.
    - labels (List[int]): A list of integers representing labels for the MP4 files.
    - num_to_class_dict (Dict[int : str]): A dictionary that maps label integers to their corresponding classes.
    - number_of_samples (int): The number of samples to display.

    Returns:
    - HTML: An HTML object displaying GIFs of the MP4 files along with their labels.
    """
    random_number = np.random.randint(0, len(files), number_of_samples)
    files = [files[i] for i in random_number]
    labels = [labels[i] for i in random_number]
    labels = [num_to_class_dict[i] for i in labels]

    html_content = ''
    for i, filename in enumerate(files):
        # Load the video
        vid = imageio.get_reader(filename, 'ffmpeg')

        # Create a BytesIO object to store the GIF
        gif_bytes = BytesIO()

        # Convert the video frames to a GIF and store it in the BytesIO object
        with imageio.get_writer(gif_bytes, format='gif', mode='I') as writer:
            for frame in vid:
                writer.append_data(frame)

        # Convert the bytes to a base64 string
        gif_base64 = base64.b64encode(gif_bytes.getvalue()).decode()

        # Create HTML to display the GIF and its label
        gif_html = f'<div style="text-align:center; display:inline-block; margin-right:20px;"><img id="gif_{i}" src="data:image/gif;base64,{gif_base64}" style="width:200px;height:200px;"><br>{labels[i]}</div>'
        html_content += gif_html

    # Create final HTML to display all GIFs and labels with infinite loop script
    final_html = f'<div style="display:flex; flex-wrap:wrap;">{html_content}</div>'
    final_html += '<script>function refreshGIFs() { document.querySelectorAll("img").forEach(img => img.src=img.src); } setInterval(refreshGIFs, 4000);</script>'

    # Display the HTML in Colab
    from IPython.display import HTML
    return HTML(final_html)


def get_mp4_files_and_labels(directory: str) -> tuple[list[str], np.ndarray]:
    """
    Retrieve a list of MP4 files and their corresponding labels from a specified directory.

    Args:
    - directory (str): A string representing the directory path containing the MP4 files.

    Returns:
    - Tuple[List[str], List[int]]: A tuple containing two lists:
        - files (List[str]): A list of strings representing file paths to MP4 files.
        - labels (np.ndarray): A np.array representing labels for the MP4 files.
    """
    files = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            files.append(directory + '/'+ file)
            label = int(file.split('-')[1][1]) - 1
            labels.append( label)
    return  files , np.array(labels)


def MP4_to_list(files: List[str], target_shape: tuple) -> List[np.array]:
    """
    Converts a list of MP4 file paths into a list of NumPy arrays representing the video frames.
    Resizes each frame to the target shape before appending.

    Parameters:
    - files (List[str]): A list of strings representing file paths to MP4 files.
    - target_shape (tuple): Target shape for resizing frames, e.g., (height, width).

    Returns:
    - List[np.array]: A list containing NumPy arrays representing the video frames.
    """
    video_list = []
    for file in files:
        frames = []
        cap = cv2.VideoCapture(file)
        ret = True
        while ret:
            ret, img = cap.read()
            if ret:
                # Convert pixel values to float32 and normalize
                img = img.astype(np.float32) / 255.0  
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(img_rgb, target_shape)
                frames.append(resized_img)
        video_list.append(np.stack(frames, axis=0))
    return video_list


def pad_videos(train_MP4_arrayed: list, max_frames: int, custom_value: int) -> np.ndarray:
    """
    Pads the input list of videos with frames containing a custom value to ensure all videos have the same number of frames.

    Parameters:
        train_MP4_arrayed (list): List of numpy arrays representing videos. Each array shape should be (num_frames, height, width, 3).
        max_frames (int): The maximum number of frames desired for all videos after padding.
        custom_value (int): The value to use for padding frames.

    Returns:
        numpy.ndarray: Array of padded videos with shape (num_videos, max_frames, height, width, 3).
    """
    padded_videos = []
    for video in train_MP4_arrayed:
        num_frames = video.shape[0]
        if num_frames < max_frames:
            num_frames_to_pad = max_frames - num_frames
            padded_frames = np.full((num_frames_to_pad, *video.shape[1:]), custom_value, dtype=np.uint8)
            padded_video = np.concatenate((video, padded_frames), axis=0)
            padded_videos.append(padded_video)
        else:
            padded_videos.append(video)
    return np.array(padded_videos)

def onehot_encode_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encodes an array of labels using TensorFlow's Keras utilities.

    Parameters:
    - y (np.ndarray): Input array of labels.
    - num_classes (int): Number of classes for one-hot encoding.

    Returns:
    - np.ndarray: One-hot encoded array of labels.
    """
    # Ensure y is integer type
    y = y.astype(np.int32)
    # Perform one-hot encoding
    encoded_labels = to_categorical(y, num_classes=num_classes)
    return encoded_labels

def make_dataset(x : np.ndarray , y : np.ndarray , num_classes: int , shuffle : bool) -> tuple[np.ndarray, np.ndarray]:
    """
    returns x , y as tuple dataset.

    Parameters:
    - x (np.ndarray): Input array .
    - y (np.ndarray): Label array
    - num_classes (int): Number of classes for one-hot encoding.
    - shuffle (bool): whether to shuffle or not

    Returns:
    - input and one_hotted labels(x , y)
      - np.ndarray: Input array
      - np.ndarray: One-hot encoded array of labels.
    """
    y = onehot_encode_labels(y ,num_classes)
    if shuffle:
      indices = np.arange(len(x))
      np.random.shuffle(indices)

      # Shuffle x and y arrays
      x = x[indices]
      y = y[indices]
      
    return (x , y)

def data_generator(x: np.ndarray, y: np.ndarray, batch_size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate batches of data for training a machine learning model.

    Parameters:
        x_train (np.ndarray): Input data array containing features.
        y_train (np.ndarray): Target data array containing labels.
        batch_size (int): Size of each batch to yield.

    Yields:
        tuple: A tuple containing a batch of input features and their corresponding target labels.
    """
    num_samples = len(x)
    while True:
        for i in range(0, num_samples, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            yield x_batch, y_batch
