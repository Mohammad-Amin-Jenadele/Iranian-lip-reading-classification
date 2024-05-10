# Package import
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
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
    final_html += '<script>function refreshGIFs() { document.querySelectorAll("img").forEach(img => img.src=img.src); } setInterval(refreshGIFs, 3000);</script>'

    # Display the HTML in Colab
    from IPython.display import HTML
    return HTML(final_html)


def get_mp4_files_and_labels(directory: str) -> tuple[list[str], list[int]]:
    """
    Retrieve a list of MP4 files and their corresponding labels from a specified directory.

    Args:
    - directory (str): A string representing the directory path containing the MP4 files.

    Returns:
    - Tuple[List[str], List[int]]: A tuple containing two lists:
        - files (List[str]): A list of strings representing file paths to MP4 files.
        - labels (List[int]): A list of integers representing labels for the MP4 files.
    """
    files = []
    labels = []
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            files.append(directory + file)
            label = int(file[4]) - 1
            labels.append( label)
    return  files , labels

