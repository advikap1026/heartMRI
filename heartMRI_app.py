import streamlit as st
from joblib import load 
import numpy as np
import opencv
import tensorflow as tf
import tensorflow as tf
import nibabel as nib
import matplotlib.pylab as plt
from tensorflow.keras.layers import Conv2D, concatenate, MaxPool2D, UpSampling2D
from tempfile import NamedTemporaryFile
import numpy as np
import os
from PIL import Image

def show_heart_label_pred_sbs(heart, pred):
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.imshow(heart.squeeze()) #, cmap = 'gray'
    a2.imshow(pred.squeeze())
    a1.set_title("heart")
    a2.set_title("predicted label")

    st.pyplot(fig)

def preprocess_nii(file):
    # Save the uploaded file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    # Load the .nii file from the temporary location
    img = nib.load(temp_file_path).get_fdata()

    # Delete the temporary file
    os.unlink(temp_file_path)

    images = []
    for i in range(img.shape[2]):
        img_slice = cv2.resize(img[:,:,i], (80, 80))
        img_slice = img_slice[:,:, np.newaxis]
        images.append(img_slice)
    return np.array(images)



model = tf.keras.models.load_model("heartModel", compile = False)

st.title('Detecting Cardiac Abnormalties in the Heart')
st.header('Advika Prashanth')
st.subheader('An easily accesable way to detect abnormalties in the heart using machine learning algorithms and cardiac imaging techniques')

st.write('Heres an example of the U_Net model on a Heart MRI')

imagee = Image.open('/content/heartExample.jpeg')
st.image(imagee)

uploaded_file = st.file_uploader("Upload Image (nii file)")
if uploaded_file is not None:

    images = preprocess_nii(uploaded_file)

    st.write('Results for first image!')
    img = np.expand_dims(images[0],0)
    pred_labels = model.predict(img) > 0.5
    show_heart_label_pred_sbs(img, pred_labels)

    pass
