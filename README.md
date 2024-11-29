# Sentiment Analysis on Twitter Data Using Machine Learning

This repository contains the code, data, and methodologies used for sentiment analysis on Twitter data. The project focuses on classifying tweets into positive, negative, or neutral sentiments using machine learning and natural language processing (NLP) techniques.

**Overview**

Twitter sentiment analysis provides organizations with real-time insights into public opinions, trends, and brand sentiment. This project implements a comprehensive pipeline that includes data preprocessing, feature extraction, and sentiment classification. It evaluates multiple classifiers to determine the most effective model for this task.

**Key Features**

**Data Preprocessing:** Cleaning and standardizing text through tokenization, lemmatization, and removal of mentions, URLs, emojis, punctuation, and stopwords.
**Feature Extraction:** Utilizes Bag-of-Words (BoW) and GloVe embeddings for feature representation.
**Classification Models:** 
Logistic Regression
Support Vector Machine (SVM)
Long Short-Term Memory (LSTM) Neural Networks
**Evaluation Metrics:** Accuracy, F1-score, Precision, and Recall across multiple test datasets.

**Technologies Used**

**Programming Language:** Python
**Libraries and Frameworks:**
PyTorch for LSTM model implementation
Scikit-learn for Logistic Regression and SVM
NLTK for text preprocessing
NumPy and Pandas for data manipulation
Matplotlib for visualization

**Results**

GloVe embeddings with LSTM showed the highest F1-score and accuracy, outperforming traditional models like Logistic Regression and SVM.
Detailed performance metrics can be found in the results/ folder.

**Visualizations**

**Sentiment Distribution:** Pie and bar charts for training and test datasets.
**Model Performance:** Comparative analysis of classifiers using precision, recall, and F1-score metrics.

**Future Enhancements**

Incorporate transformer-based models like BERT for improved context understanding.
Expand dataset to include multi-lingual tweets for broader applicability.
Optimize LSTM performance through hyperparameter tuning.

**Contact**

**For queries, reach out to:**

Author: Rifhath Aslam Jageer Hussain
Email: rifhathaslam.jr.162@gmail.com
LinkedIn: [Rifhath Aslam](https://www.linkedin.com/in/rifhath-aslam-j-791a6a21b/)
