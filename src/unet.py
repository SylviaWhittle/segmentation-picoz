"""A U-NET model for segmentation of atomic force microscopy image grains."""

# import adam optimizer
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
import tensorflow as tf
import numpy as np


# DICE Loss
def dice_loss(y_true, y_pred, smooth=1e-5):
    """DICE loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    dice : tf.Tensor
        The DICE loss.
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice


# Weighted DICE Loss
def weighted_dice_loss(y_true, y_pred, y_weights, smooth=1e-5):
    """Weighted DICE loss function that emphasizes certain regions.

    Parameters
    ----------
    y_true : tf.Tensor
        True values with shape (batch_size, height, width, channels).
    y_pred : tf.Tensor
        Predicted values with shape (batch_size, height, width, channels).
    y_weights : tf.Tensor
        Weight map with shape (batch_size, height, width) or (batch_size, height, width, 1).
        Higher values indicate regions of greater importance.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    dice : tf.Tensor
        The weighted DICE loss.
    """
    # Ensure weights have the right shape - expand dims if needed
    if len(y_weights.shape) == 3:  # (batch_size, height, width)
        y_weights = tf.expand_dims(y_weights, axis=-1)
    
    # Ensure all tensors are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_weights = tf.cast(y_weights, tf.float32)
    
    # Apply weights to the intersection and union calculations
    weighted_intersection = tf.reduce_sum(y_weights * y_true * y_pred, axis=(1, 2))
    weighted_sum_pred = tf.reduce_sum(y_weights * tf.square(y_pred), axis=(1, 2))
    weighted_sum_true = tf.reduce_sum(y_weights * tf.square(y_true), axis=(1, 2))
    
    dice = 1 - (2 * weighted_intersection + smooth) / (weighted_sum_pred + weighted_sum_true + smooth)
    return dice


# Custom Loss Class for Weighted DICE
class WeightedDiceLoss(tf.keras.losses.Loss):
    """Custom Keras loss class for weighted DICE loss.
    
    This allows you to pass weight maps along with your training data.
    Use it by concatenating weight maps as an additional channel to y_true.
    """
    
    def __init__(self, smooth=1e-5, name="weighted_dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        """
        Expects y_true to have shape (batch_size, height, width, channels + 1)
        where the last channel contains the weight map.
        y_pred should have shape (batch_size, height, width, channels).
        """
        # Split y_true into actual labels and weights
        actual_y_true = y_true[..., :-1]  # All channels except last
        y_weights = y_true[..., -1:]      # Last channel is weights
        
        return weighted_dice_loss(actual_y_true, y_pred, y_weights, self.smooth)


# IoU Loss
def iou_loss(y_true, y_pred, smooth=1e-5):
    """Intersection over Union loss function.

    Parameters
    ----------
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    smooth : float
        Smoothing factor to prevent division by zero.

    Returns
    -------
    iou : tf.Tensor
        The IoU loss.
    """
    # Ensure the tensors are of the same shape
    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape[-1] == 1 else y_true
    y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape[-1] == 1 else y_pred
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_of_squares_pred = tf.reduce_sum(tf.square(y_pred), axis=(1, 2))
    sum_of_squares_true = tf.reduce_sum(tf.square(y_true), axis=(1, 2))
    iou = 1 - (intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true - intersection + smooth)
    return iou


def unet_model(
    image_height,
    image_width,
    image_channels,
    output_classes,
    learning_rate,
    conv_activation_function,
    final_activation_function,
    loss_function,
):
    """U-NET model definition function.

    Parameters
    ----------
    image_height : int
        Image height.
    image_width : int
        Image width.
    image_channels : int
        Number of image channels.
    output_classes : int
        Number of output classes.
    learning_rate : float
        Learning rate for the Adam optimizer.
    conv_activation_function : str
        Activation function to use in the convolutional layers.
    final_activation_function : str
        Activation function to use in the final layer.
    loss_function : str
        Loss function to use in the model.

    Returns
    -------
    model : keras.models.Model
        Single channel U-NET model for segmentation.
    """

    inputs = Input((image_height, image_width, image_channels))

    # Downsampling
    # Downsample with increasing numbers of filters to try to capture more complex features (first argument)
    # Dropout is used to try to prevent overfitting. Increase if overfitting happens.
    # Dropout increases deeper into the model to further help prevent overfitting.

    conv1 = Conv2D(
        16, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(
        16, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv1)
    pooled1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(
        32, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(
        32, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv2)
    pooled2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(
        64, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(
        64, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv3)
    pooled3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(
        128, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(
        128, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv4)
    pooled4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        256, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(pooled4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(
        256, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv5)

    # Upsampling
    # Conv2DTranspose is used as a sort of inverse convolution, to upsample the image
    # A concatenation is used to force context from the original image, providing information about what context a
    # feature stems from.

    up6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(
        128, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(
        128, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv6)

    up7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(
        64, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(
        64, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv7)

    up8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(
        32, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(
        32, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv8)

    up9 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(
        16, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(
        16, kernel_size=(3, 3), activation=conv_activation_function, kernel_initializer="he_normal", padding="same"
    )(conv9)

    # Make predictions of classes based on the culminated data
    outputs = Conv2D(output_classes, kernel_size=(1, 1), activation=final_activation_function)(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # print structure
    print(type(model))
    # custom learning rate
    optimizer = Adam(learning_rate)

    # Loss function
    if loss_function == "dice_loss":
        # loss = dice_loss
        # loss = tf.keras.losses.Dice(reduction="sum_over_batch_size", name="dice")
        loss = dice_loss
    elif loss_function == "weighted_dice_loss":
        # Use the custom WeightedDiceLoss class
        loss = WeightedDiceLoss()
        print("Using WeightedDiceLoss - remember to concatenate weight maps to y_true as the last channel")
    elif loss_function == "iou_loss":
        loss = iou_loss
    elif loss_function == "binary_crossentropy":
        loss = "binary_crossentropy"
    elif loss_function == "categorical_crossentropy":
        loss = "categorical_crossentropy"
    elif loss_function == "sparse_categorical_crossentropy":
        loss = "sparse_categorical_crossentropy"
    elif loss_function == "mean_squared_error":
        loss = "mean_squared_error"
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    return model
