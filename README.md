# Iranian Lip Reading Classification

Lip reading is an intriguing field that sits at the crossroads of computer vision and natural language processing, serving as a vital means to bridge communication barriers, especially for the hearing impaired. This project focuses on the challenging task of Iranian lip reading classification, employing deep learning techniques.

## Introduction

In this Repo, we delve into the task of classifying Iranian lip movements captured in MP4 files. The dataset consists of both training and test sets, with each instance assigned one of the following labels:
- Iran
- Khoshhal (Happy)
- Moa'lem (Teacher)
- Salam (Hello)
- Khodahafez (Bye)

We explore various deep learning architectures to accomplish this task, including:

1. **CNN and RNN**: Combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to capture spatial and temporal features from the lip movements.
2. **Video Vision Transformer (ViVit)**: Experimenting with a Transformer architecture combined with CNNs, leveraging self-attention mechanisms for lip reading.
3. **3D Convolution**: Utilizing 3D convolutional networks to learn spatial and temporal patterns simultaneously from the video sequences.

## Results

### CNN and RNN
- Achieved an accuracy of 85% on the test set.

### ViVit Model
- Scored an accuracy of 80% on the test set.
- Despite having more parameters than CNN-RNN, the performance was affected due to padding frames and the small dataset size.

### Conv3D Model
- Impressive accuracy of 92% on the test set.
- Challenges included hyperparameter tuning and memory consumption issues.

## Conclusion

While each model exhibited strengths and weaknesses, the Conv3D model emerged as the most promising with its high accuracy. However, optimizing its performance requires careful parameter tuning and addressing memory constraints. This project underscores the potential of deep learning in Iranian lip reading classification while highlighting areas for further improvement.

The ViVit model works well with big datasets for capturing temporal connections. CNN and RNN models are strong when temporal connections span at least 40 steps. If it's more than 40 steps, go for ViVit or Conv3D models. ViVit needs a large dataset, while Conv3D may consumes more RAM and time for tuning but works well with small dataset.

## Let's Connect

Feel free to explore the notebook and contribute to advancing Iranian lip reading classification!

---
