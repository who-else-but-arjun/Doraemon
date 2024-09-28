from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
import math
import os 
import cv2
def model():
    
    SRCNN = Sequential()
    
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    #input_shape takes image of any height and width as long it is one channel
    #that is how the SRCNN handles input,it handles image slice inputs, it doesn't work at all 3 channels at once
    #SRCNN was trained on the luminescence channel in the YCrCb color space 
    
    adam = Adam(learning_rate=0.0001)
  
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    SRCNN.load_weights('3051crop_weight_200.h5')
    return SRCNN

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
    
    return img

def postprocess_image(pred):
    pred = np.squeeze(pred)
    pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    
    return pred

def enhance_images(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):  
            image_path = os.path.join(input_folder, filename)
            processed_img = preprocess_image(image_path)
            pred = model.predict(processed_img)
            enhanced_img = postprocess_image(pred)
            output_path = os.path.join(output_folder, f"enhanced_{filename}")
            cv2.imwrite(output_path, enhanced_img)
            print(f"Enhanced image saved: {output_path}")
