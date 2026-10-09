# 🌍 Disaster Tweet Classification & Sentiment Analysis

An NLP-based machine learning application that classifies tweets as disaster-related or non-disaster-related using a fine-tuned DistilBERT model. It also analyzes the emotional tone of text using VADER sentiment analysis.

## 🚀 Live Demo

**Streamlit App:** [Launch Disaster Tweet Analyzer](https://project7-disaster-tweet-classification-sentiment-analysis-git.streamlit.app/)

Try entering a tweet to view its predicted disaster classification, model confidence, and sentiment score.

## 🎯 Project Objective

The objective is to develop an NLP application that distinguishes tweets reporting real disaster-related events from ordinary text, while independently analyzing the sentiment expressed in each tweet.

## ✨ Key Features

- **Disaster Classification:** Predicts whether a tweet is disaster-related or non-disaster-related.
- **Transformer-Based NLP:** Uses a fine-tuned DistilBERT model for text classification.
- **Prediction Confidence:** Displays the model's confidence in its prediction.
- **Sentiment Analysis:** Uses VADER to identify positive, negative, or neutral sentiment.
- **Interactive Web App:** Provides an easy-to-use interface built with Streamlit.
- **Sample Tweets:** Includes examples for testing the application.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- DistilBERT
- NLTK and VADER
- Streamlit
- Git and Git LFS

## 🔄 Project Workflow

1. Load and explore the disaster tweet dataset.
2. Clean and preprocess the text data.
3. Explore text vectorization approaches, including Bag of Words and TF-IDF.
4. Fine-tune DistilBERT for disaster tweet classification.
5. Evaluate the model using classification metrics.
6. Integrate VADER for sentiment analysis.
7. Develop the Streamlit application.
8. Version-control the project using GitHub and Git LFS.
9. Deploy the application using Streamlit Community Cloud.

## 📊 Model Evaluation

The reported test results for the DistilBERT model are:

| Metric | Test Result |
|---|---:|
| Accuracy | 84.12% |
| Precision | 81.99% |
| Recall | 80.73% |
| F1-score | 81.36% |

These metrics summarize the model's performance on the test dataset. Predictions on individual tweets may differ, especially for ambiguous text.

## 🧠 How It Works

**Disaster Classification:** The tweet is tokenized and passed to the fine-tuned DistilBERT model, which predicts one of two classes.

**Sentiment Analysis:** VADER calculates a sentiment score that is mapped to positive, negative, or neutral sentiment.

The two analyses are independent. A disaster-related tweet can express positive, negative, or neutral sentiment.

## 📁 Repository Structure

```text
disaster-tweet-classification-sentiment-analysis/
├── distilbert_disaster_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── app.py
├── requirements.txt
├── .gitattributes
├── .gitignore
├── README.md
└── LICENSE
```

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/kgayattre96/Project_7-Disaster-Tweet-Classification-Sentiment-Analysis-.git
cd Project_7-Disaster-Tweet-Classification-Sentiment-Analysis-
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Git LFS if it is not already installed, then retrieve the model files:

```bash
git lfs install
git lfs pull
```

Install the Python packages:

```bash
python -m pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run app.py
```

## ⚠️ Limitations

- Tweets containing disaster-related keywords may be misclassified when the context is fictional or unrelated to a real event.
- Low-confidence predictions should be reviewed manually.
- Sentiment scores represent linguistic sentiment, not confirmation of the seriousness or reality of an event.
- Model performance on unseen text may differ from test-set performance.

## 🔮 Future Improvements

- Improve handling of ambiguous and context-dependent tweets.
- Evaluate performance across different disaster categories.
- Expand error analysis and test with more real-world examples.
- Explore explainability techniques to better understand model predictions.

## 👩‍💻 Author

Developed as an NLP and machine learning project demonstrating text classification, transformer-based modeling, sentiment analysis, and application deployment.