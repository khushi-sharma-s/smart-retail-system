    import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 🛍️ WEBSITE BASIC LAYOUT
# ==========================================
st.title("🛍️ Smart Retail Analytics System")

# Sidebar navigation for 3 modules
option = st.sidebar.selectbox("Choose a Module", [
    "Home", 
    "ANN - Sales Prediction", 
    "CNN - Product Image Classification", 
    "RNN - Review Sentiment Analysis"
])

# ==========================================
# 🏠 MODULE 1: HOME PAGE
# ==========================================
if option == "Home":
    st.write("Welcome to your Mixed Deep Learning Project (ANN + CNN + RNN).")
    st.write("This system uses 3 different Deep Learning models to manage an E-Commerce Store.")

# ==========================================
# 📊 MODULE 2: ANN - SALES PREDICTION
# ==========================================
elif option == "ANN - Sales Prediction":
    st.header("📊 Walmart Weekly Sales Prediction")
    st.write("Enter the store details below to predict the weekly sales.")

    # 1. Load the trained ANN model
    try:
        ann_model = tf.keras.models.load_model('ann_sales_model.keras')
    except:
        st.error("Error: 'ann_sales_model.keras' not found in folder.")

    # 2. Create User Input Fields
    store = st.number_input("Store Number (1-45)", min_value=1, max_value=45, value=1)
    holiday = st.selectbox("Is it a Holiday Week?", ["No", "Yes"])
    temp = st.number_input("Temperature (in Fahrenheit)", value=60.0)
    fuel = st.number_input("Fuel Price", value=3.5)
    cpi = st.number_input("CPI (Consumer Price Index)", value=211.0)
    unemp = st.number_input("Unemployment Rate", value=7.0)
    month = st.slider("Month", min_value=1, max_value=12, value=1)
    year = st.selectbox("Year", [2010, 2011, 2012])

    holiday_flag = 1 if holiday == "Yes" else 0

    # 3. Predict Button
    if st.button("Predict Weekly Sales"):
        # Format the inputs exactly how the network expects it
        input_data = np.array([[store, holiday_flag, temp, fuel, cpi, unemp, month, year]])
        prediction = ann_model.predict(input_data)
        st.success(f"💰 Estimated Weekly Sales for this store: ${prediction[0][0]:,.2f}")

# ==========================================
# 🖼️ MODULE 3: CNN - PRODUCT IMAGE CLASSIFICATION
# ==========================================
elif option == "CNN - Product Image Classification":
    st.header("🖼️ Fashion Product Categorization")
    st.write("Upload a clothing/footwear item image (28x28 grayscale) to identify its category.")

    # 1. Load CNN Model
    cnn_model = tf.keras.models.load_model('cnn_product_model.keras')
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 2. Image Upload Box
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        # 3. Preprocess the image to match Fashion MNIST format
        img_gray = image.convert('L').resize((28, 28))
        img_array = np.array(img_gray) / 255.0
        img_input = img_array.reshape(1, 28, 28, 1) 
        
        if st.button("Classify Image"):
            pred = cnn_model.predict(img_input)
            predicted_class = class_names[np.argmax(pred)]
            st.success(f"👗 Predicted Item Category: **{predicted_class}**")

# ==========================================
# 📝 MODULE 4: RNN - REVIEW SENTIMENT ANALYSIS
# ==========================================
elif option == "RNN - Review Sentiment Analysis":
    st.header("📝 Customer Review Sentiment Analysis")
    st.write("Type a customer review below to check if the feedback is Positive or Negative.")

    # 1. Load RNN Model and saved Tokenizer
    rnn_model = tf.keras.models.load_model('rnn_sentiment_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    # 2. Review Text Input
    user_review = st.text_area("Enter Customer Review here:")
    
    if st.button("Analyze Sentiment"):
        if user_review.strip() == "":
            st.warning("Please type something first!")
        else:
            # 3. Preprocess user text just like we did during training
            seq = loaded_tokenizer.texts_to_sequences([user_review])
            padded_seq = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
            
            score = rnn_model.predict(padded_seq)
            
            # 4. Display Final Sentiment Label
            if score > 0.5:
                st.success(f"😊 **Positive Feedback** (Confidence: {score[0][0]*100:.2f}%)")
            else:
                st.error(f"😞 **Negative Feedback** (Confidence: {(1-score[0][0])*100:.2f}%)")

