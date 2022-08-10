import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.utils import img_to_array


@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model(r"C:\Users\amit_kumar\project\model.h5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Image Classification Model
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png", "jpeg"])

import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        #     # convert to array
        # img = img_to_array(image_data)
        # # reshape into a single sample with 3 channels
        # img = img.reshape(1, 32, 32, 3)
        # # prepare pixel data
        # img = img.astype('float32')
        # img = img / 255.0
    
        size = (32,32)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS) 
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(32, 32),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    score = tf.nn.softmax(prediction[0])
    class_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    st.write(class_names)
    st.write(prediction)
    st.write(score)
    st.write("""
         ### PREDICTION
         """
         )
    st.write(class_names[np.argmax(score)])
    st.write("""
         ### SCORE
         """
         )
    st.write(100*np.max(score))
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)