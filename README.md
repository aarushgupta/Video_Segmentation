# Video_Segmentation
Segment videos into groups of frames which represent a common human action. Sample video and corresponding frames have been provided.

## Summary
The frames of the sample video are fed into a pre-trained Keras model of **VGGNet** to extract the features of the frames.
The extracted features of the frames have been used for **Spectral Clustering** of the frames using the *Normalized Cuts algorithm*.

## Prerequisites
1. Anaconda
2. Scikit-Learn
3. Keras alongwith Theano or Tensorflow(recommended) 

## How to use
1. Set the value of `k` to the desired number of clusters.
2. Pass the number of frames, format of the frames and the path, where the frames are located, in the `get_features` function
3. Pass the number of frames in the `adjacency_matrix` function
4. Run the rest of the code

## How to extract the frames of the video
Use OpenCV

## To-Do
1. Add Steps to extract the frames of the video
2. Implement Oversegmentation and add Conv3D model for finer segmentation.
