import os
import numpy as np
from keras.layers import Input, Activation, Add, UpSampling2D, Conv2D, Lambda
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from layer_utils import ReflectionPadding2D, res_block
from tensorflow_addons.layers import InstanceNormalization 

channel_rate = 64
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)

ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
n_blocks_gen = 18

def generator_model():
    """Build generator architecture."""
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = InstanceNormalization()(x) 
        x = Activation('relu')(x)

    mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
        x = InstanceNormalization()(x) 
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)
    outputs = x  

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model

def load_generator_weights(generator, weights_path):
    """Load pre-trained weights into the generator model."""
    generator.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return generator

def preprocess_image(image_path, image_shape=(256, 256)):
    """Preprocess the image for the model."""
    image = load_img(image_path, target_size=image_shape)
    image = img_to_array(image)
    image = (image / 127.5) - 1  
    return np.expand_dims(image, axis=0)

def postprocess_image(deblurred_image):
    """Post-process the deblurred image from the model output."""
    deblurred_image = (deblurred_image + 1) * 127.5 
    deblurred_image = np.clip(deblurred_image, 0, 255).astype('uint8')
    return array_to_img(deblurred_image[0])

def deblur_image(input_folder, output_folder, weights_path):
    """Deblur all images in the input folder using the generator model."""
    generator = generator_model()
    generator = load_generator_weights(generator, weights_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            image = preprocess_image(image_path)
            deblurred_image = generator.predict(image)
            result_image = postprocess_image(deblurred_image)
            output_path = os.path.join(output_folder, filename)
            result_image.save(output_path)
            print(f"Saved deblurred image to {output_path}")

if __name__ == '__main__':
    input_folder = 'image'      
    output_folder = 'image_out'  
    weights_path = 'generator.h5'

    deblur_image(input_folder, output_folder, weights_path)
