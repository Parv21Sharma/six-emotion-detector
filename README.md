# 🧠 Six Emotion Detector

A machine learning and deep learning based **emotion classification system** that predicts the emotion expressed in a given sentence.

The application compares predictions from **four different models** and displays their predicted emotion along with confidence/decision scores.

## 🚀 Live Demo

Try the deployed application here:

https://six-emotion-detector-2lwczt3wqjhcju9bkrtgfv.streamlit.app/

## 🎯 Emotions

The system classifies text into six emotions:

- 😠 Anger
- 😨 Fear
- 😊 Joy
- ❤️ Love
- 😢 Sadness
- 😲 Surprise

## 🤖 Models Used

Four different classification models are implemented:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. Long Short-Term Memory (LSTM)

The application runs all four models on the same input sentence and provides a model comparison.

## 📊 Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | ~82.9% |
| Random Forest | ~84.8% |
| Support Vector Machine | ~81.9% |
| LSTM | ~86.4% |

The LSTM achieved the highest test accuracy among the implemented models.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TensorFlow / Keras
- Streamlit
- TF-IDF
- LSTM
- GitHub
- Git LFS

## 🔄 Text Preprocessing

The text is processed using:

- Removal of non-alphabetic characters
- Conversion to lowercase
- Stopword removal
- Porter stemming

For the traditional machine learning models, the cleaned text is converted into numerical features using **TF-IDF**.

For the LSTM model, the cleaned text is converted into integer sequences and padded to a fixed length.

## 🧠 Model Comparison

The Streamlit application allows the user to enter a sentence and receive predictions from all four models simultaneously.

For example:

**Input:**  
> I am really happy today!

The application displays:

- Logistic Regression prediction and confidence
- Random Forest prediction and confidence
- SVM prediction and decision score
- LSTM prediction and confidence
- Overall model consensus

## 📁 Project Structure

```text
six-emotion-detector/
│
├── models/
│   ├── logistic_regression/
│   │   └── logistic_regression.pkl
│   │
│   ├── random_forest/
│   │   └── random_forest.pkl
│   │
│   ├── svm/
│   │   └── svm.pkl
│   │
│   ├── lstm/
│   │   ├── lstm_model.keras
│   │   └── vocab_info.pkl
│   │
│   ├── label_encoder.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py
├── requirements.txt
├── sarcasmdata.txt
├── NLP Six Emotion Classification.ipynb
└── README.md
