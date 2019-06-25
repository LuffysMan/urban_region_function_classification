import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D, 
    MaxPooling2D, 
    AveragePooling2D, 
    Dense, 
    Flatten, 
    Activation
)


class PlainNetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
        Returns:
            The keras `Model`.
        """
        # Load function from str if needed.
        inputs = Input(input_shape)

        # conv1 = Conv2D(64, 3, strides=(1,1), padding='SAME', activation='relu',
        #                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(inputs)
        # pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        # conv2 = Conv2D(128, 3, strides=(1,1), padding='SAME', activation='relu',
        #                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(pool1)
        # pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

        # conv3 = Conv2D(256, 3, strides=(1,1), padding='SAME', activation='relu',
        #                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))(pool2)
        # pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
        pool3_flat = tf.keras.layers.Flatten()(inputs)

        dense1 = tf.keras.layers.Dense(256, activation='relu')(pool3_flat)
        dense2 = tf.keras.layers.Dense(num_outputs, activation='softmax')(dense1)

        model = Model(inputs=inputs, outputs=dense2)
        return model

    

