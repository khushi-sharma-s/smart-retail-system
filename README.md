# 🛍️ Smart Retail AI System: End-to-End E-Commerce Solution
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) 
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg) 
![Machine Learning](https://img.shields.io/badge/Deep%20Learning-ANN%20%7C%20CNN%20%7C%20RNN-success.svg)
## 📌 Project Overview
The **Smart Retail AI System** is a comprehensive Deep Learning project designed to solve three major challenges in the modern e-commerce and retail industry. By leveraging different neural network architectures (ANN, CNN, and RNN), this project provides a unified solution for sales forecasting, automated product categorization, and customer sentiment analysis.
This system is deployed as an interactive web application using **Streamlit**, making it highly accessible for end-users to test the models in real-time.
## 🚀 Key Features & Models Developed 
### 1. Sales Forecasting (Artificial Neural Network - ANN)
* **Objective:** Predict weekly store sales based on historical data, temperature, fuel prices, CPI, and unemployment rates.
* **Dataset:** Walmart Store Sales Dataset.
* **Architecture:** Multi-layered Perceptron (MLP) / ANN with standardized numerical inputs.
* **Impact:** Helps retail managers optimize inventory and anticipate revenue.
* ### 2. Product Image Classification (Convolutional Neural Network - CNN) * **Objective:** Automatically categorize apparel and clothing items from images.
* **Dataset:** Fashion MNIST (10 categories including T-shirts, Trousers, Sneakers, etc.).
* **Architecture:** CNN with `Conv2D` and `MaxPooling2D` layers for feature extraction, followed by dense layers.
* **Impact:** Automates inventory cataloging and enhances the visual search experience for users.
### 3. Customer Sentiment Analysis (Recurrent Neural Network - RNN/LSTM)
* **Objective:** Analyze text reviews left by customers and classify them as positive or negative. *  * **Dataset:** Women's E-Commerce Clothing Reviews Dataset.
* **Architecture:** Natural Language Processing (NLP) pipeline using text tokenization, padding, Word Embeddings, and an LSTM (Long Short-Term Memory) network.
* **Impact:** Allows businesses to automatically gauge customer satisfaction and flag negative feedback for quick resolution.
## 🛠️ Technology Stack
* **Languages:** Python
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Data Visualization:** Matplotlib
* **Web Deployment:** Streamlit
* **NLP Tools:** Keras Tokenizer, Pad Sequences
## 📂 Project Structure 
```text
📦 Smart_Retail_Project
┣ 📜 smart_retail.py # Main Streamlit application script
┣ 📜 e-commerceproject.ipynb # Jupyter Notebook with EDA and Model Training
┣ 📜 ann_sales_model.keras # Trained ANN model for sales prediction
┣ 📜 cnn_product_model.keras # Trained CNN model for image classification
┣ 📜 rnn_sentiment_model.keras # Trained RNN/LSTM model for text analysis
┣ 📜 tokenizer.pickle # Saved NLP tokenizer for text preprocessing
┗ 📜 requirements.txt # Project dependencies
## 💻 How to Run Locally

 **Clone the repository:**
   ```bash
   git clone [https://github.com/khushi-sharma-s/smart-retail-system.git](https://github.com/khushi-sharma-s/smart-retail-system.git)
   cd smart-retail-system
pip install -r requirements.txt
streamlit run Smart_Retail_Project/smart_retail.py
