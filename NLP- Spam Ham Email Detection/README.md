# 📧 Email Spam Detection with NLP

A machine learning project to detect spam vs ham emails using NLP techniques and Naive Bayes, with a Streamlit web app for deployment.

## 🚀 Features
- EDA and model building in Jupyter notebook
- Text preprocessing: lowercasing, stopword removal, stemming
- Vectorization using CountVectorizer
- Model training with Multinomial Naive Bayes
- Evaluation with confusion matrix and classification report
- Streamlit web app for interactive prediction

## ☁️ Deploy to Streamlit Cloud
- Enter any email text in the web app and click **Predict**.
- See whether it’s predicted as "🚫 Spam" or "✅ Not Spam".
Ham
(!ham)[ham.png]

## 📄 Dataset
The dataset should be `emails.csv` with:
- `text`: email content
- `spam`: label (1 for spam, 0 for ham)

## 📊 Model & EDA
- Based on CountVectorizer and Multinomial Naive Bayes.
- Explored data distribution, trained model, and evaluated accuracy.


