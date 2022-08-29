#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                     Reference:                      #
#   Olaf Ronneberger, Philipp Fischer, Thomas Brox.   #
#                    18 May 2015.                     #
#          U-Net: Convolutional Networks for          #
#            Biomedical Image Segmentation.           #
#                    MICCAI 2015.                     #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture
import tensorflow_addons as tfa

#-----------------------------------------------------#
#         Architecture class: U-Net Standard          #
#-----------------------------------------------------#
""" The Standard variant of the popular U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D U-Net standard model using Keras
    create_model_3D:        Creating the 3D U-Net standard model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_filters=32, depth=4, activation='softmax',
                 in_normalization=True):
        # Parse parameter
        self.n_filters = n_filters
        self.depth = depth
        self.activation = activation
        # Batch normalization settings
        self.in_norm = in_normalization

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2**i
            cnn_chain, last_conv = contracting_layer_2D(cnn_chain, neurons,
                                                        self.in_norm)
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2**self.depth
        cnn_chain = middle_layer_2D(cnn_chain, neurons, self.in_norm)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2**i
            cnn_chain = expanding_layer_2D(cnn_chain, neurons,
                                           contracting_convs[i], self.in_norm)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1),
                   activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2**i
            cnn_chain, last_conv = contracting_layer_3D(cnn_chain, neurons,
                                                        self.in_norm
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2**self.depth
        cnn_chain = middle_layer_3D(cnn_chain, neurons, self.in_norm)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2**i
            cnn_chain = expanding_layer_3D(cnn_chain, neurons,
                                           contracting_convs[i], self.in_norm)

        # Output Layer
        conv_out = Conv3D(n_labels, (1, 1, 1),
                   activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

#-----------------------------------------------------#
#                   Subroutines 2D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_2D(input, neurons, in_norm):
    conv1 = Conv2D(neurons, (3,3), activation='relu', padding='same')(input)
    if in_norm : conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv2 = Conv2D(neurons, (3,3), activation='relu', padding='same')(conv1)
    if in_norm : conv2 = tfa.layers.InstanceNormalization()(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_2D(input, neurons, in_norm):
    conv_m1 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(input)
    if in_norm : conv_m1 = tfa.layers.InstanceNormalization()(conv_m1)
    conv_m2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv_m1)
    if in_norm : conv_m2 =tfa.layers.InstanceNormalization()(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_2D(input, neurons, concatenate_link, in_norm):
    up = concatenate([Conv2DTranspose(neurons, (2, 2), strides=(2, 2),
                     padding='same')(input), concatenate_link], axis=-1)
    conv1 = Conv2D(neurons, (3, 3,), activation='relu', padding='same')(up)
    if in_norm : conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv1)
    if in_norm : conv2 = tfa.layers.InstanceNormalization()(conv2)
    return conv2

#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_3D(input, neurons, in_norm):
    conv1 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(input)
    if in_norm : conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv2 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(conv1)
    if in_norm : conv2 = tfa.layers.InstanceNormalization()(conv2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_3D(input, neurons, in_norm):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(input)
    if in_norm : conv_m1 = tfa.layers.InstanceNormalization()(conv_m1)
    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv_m1)
    if in_norm : conv_m2 = tfa.layers.InstanceNormalization()(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_3D(input, neurons, concatenate_link, in_norm):
    up = concatenate([Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                     padding='same')(input), concatenate_link], axis=4)
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(up)
    if in_norm : conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv1)
    if in_norm : conv2 = tfa.layers.InstanceNormalization()(conv2)
    return conv2