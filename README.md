# Emotion-Detection-using-CNN

  In this project an emotion detection system was built based on Convolutional Neural Network  using Keras. Here the dataset used is CK-48. This system is trained to detect 5 emotions,  namely – anger, happiness, sadness, surprise and fear.


Various methods used 

ImageDataGenerator: Image augmentation is a technique of applying different 
transformations to original images which results in multiple transformed copies of the same 
image.

Conv2d: Conv2D is a 2D Convolution Layer, this layer creates a convolution kernel that is 
wind with layers input which helps produce a tensor of outputs. The parameters given to 
this layer are kernel size, strides, padding, activation function, etc. 

Maxpooling: Maximum pooling, or max pooling, is a pooling operation that calculates the 
maximum, or largest, value in each patch of each feature map. It is a down sampling 
strategy in Convolutional Neural Networks. It helps in reducing the dimensionality of the 
data.

Flatten: This method basically converts the currently present layer’s data into a onedimensional vector

Dense: Dense implements the operation: output = activation(dot(input, kernel) + bias) 
where activation is the element-wise activation function passed as the activation argument, 
kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.

Earlystopping - is implemented so that whenever the loss is not decreasing or is increasing, 
the iteration is stopped at that epoch. This is done so as to prevent overfitting and also to 
find the best possible accuracy and loss.

ModelCheckpoint - is used so that the model can identify and save the points where the 
best training accuracy and loss was observed. 
