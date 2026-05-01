# ISY503 Intelligent Systems — Assessment 3
## Sentiment Analysis NLP Project

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)

> A neural network-based sentiment analysis system trained on Amazon product reviews, capable of classifying user input text as **Positive** or **Negative**. Two models are implemented and compared: 
- a Bidirectional LSTM trained from scratch 
- a fine-tuned BERT transformer. 
Built as part of ISY503 Intelligent Systems at Torrens University Australia.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Results & Accuracy Comparison](#results--accuracy-comparison)
- [Ethical Considerations](#ethical-considerations)
- [Individual Contributions](#individual-contributions)

---

## Project Overview

This project implements a sentiment analysis pipeline using two approaches:

1. **Bidirectional LSTM** — trained from scratch on the JHU Multi-Domain Sentiment Dataset
2. **BERT** — Google's pre-trained transformer model fine-tuned on the same dataset

Both models accept plain text product reviews and return a binary classification: **Positive review** or **Negative review**. A web interface is provided for both models.

---

## Team Members

| Name | Student ID | GitHub | Contribution |
|------|-----------|--------|-------------|
| Nomayer Hossain | A00176827 | [@ubernomzy](https://github.com/username) | LSTM Model, BERT Model, Data Pipeline |
| Andrew Chang | [A00031568] | [@fisherfriendman](https://github.com/username) | [To be updated] |
| Kelly Thaiane Costa de Araujo | [Student ID] | [@username](https://github.com/username) | [To be updated] |

---

## Repository Structure

```
ISY503-Assessment3-NLP/
│
├── model_lstm.ipynb            # Bidirectional LSTM model (trained from scratch)
├── model_bert.ipynb            # BERT model (fine-tuned on Amazon reviews)
│
├── web_app/
│   ├── app.py                  # Flask web application (supports both models)
│   └── templates/
│       └── index.html          # Frontend interface
│
├── models/                     # Saved model weights (not tracked - see .gitignore)
│   ├── lstm_model.pt
│   └── bert_finetuned/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Source:** Multi-Domain Sentiment Dataset (Blitzer, Dredze & Pereira, 2007)
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

The dataset contains Amazon product reviews across 4 categories: Books, DVDs, Electronics, and Kitchen & Housewares. Each with positive and negative labels.

**Preprocessing applied:**
- Extracted review text from pseudo-XML format using regex
- Removed punctuation, HTML tags, stopwords; applied lemmatisation
- Removed outlier reviews under 10 or over 500 words
- Shuffled and split 80% training / 10% validation / 10% test
- Total reviews after cleaning: 7,614 (balanced: ~3,800 positive, ~3,800 negative)

> **Note on data:** Data files are not tracked in this repository due to size. Download from the link above and upload to Google Drive at: `MyDrive/Colab Notebooks/ISY503-Assessment3-NLP/Data/`

---

## Models

### Model 1 — Bidirectional LSTM (`model_lstm.ipynb`)
- Custom architecture built from scratch in PyTorch
- Embedding (38,934 vocab × 200 dim) → BiLSTM (2 layers, 128 hidden) → Dropout (0.5) → FC → Sigmoid
- Trained with early stopping, Adam optimiser, lr=0.0003
- **Test Accuracy: 76.88%**

### Model 2 — BERT Fine-tuned (`model_bert.ipynb`)
- `bert-base-uncased` pre-trained transformer from HuggingFace
- Fine-tuned on the same Amazon reviews dataset for 3 epochs
- AdamW optimiser with linear warmup scheduler, lr=2e-5
- **Test Accuracy: 89.05%**

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Google Colab (recommended) with T4 GPU enabled

### Install dependencies

```bash
pip install torch transformers nltk pandas numpy flask
```

### Or in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/ISY503-Assessment3-NLP.git
%cd ISY503-Assessment3-NLP
!pip install -r requirements.txt
```

---

## Running the Project

### Run LSTM model
Open `model_lstm.ipynb` in Google Colab and run all cells in order.

### Run BERT model
Open `model_bert.ipynb` in Google Colab and run all cells in order.

### Launch web interface
[To be updated]

---

## Results & Accuracy Comparison

| Model | Architecture | Val Accuracy | Test Accuracy |
|-------|-------------|-------------|--------------|
| Bidirectional LSTM | Custom PyTorch (from scratch) | 77.27% | 76.88% |
| BERT Fine-tuned | HuggingFace Transformer | 91.95% | 89.05% |

The significant accuracy gap demonstrates the advantage of transfer learning. BERT begins with deep language knowledge from pre-training on 3.3 billion words, while the LSTM learns entirely from the 7,614 training reviews.

---

## Ethical Considerations

**1. Dataset labelling bias**
The original positive/negative file labels are not guaranteed clean splits. The dataset owner notes reviews were randomly drawn across files. Labels were therefore derived from star ratings (≥4 stars = positive, ≤2 stars = negative) with 3-star neutral reviews excluded.

**2. Domain generalisation**
Both models were trained on product reviews only. Applying either to other domains (healthcare, legal, social media) without retraining risks misclassification and unintended consequences.

**3. Fairness and representation**
The dataset may over-represent certain product categories and demographics of reviewers, introducing bias into the model's predictions.

**4. Transparency**
Any commercial deployment of this system would require disclosure to users that sentiment is being algorithmically assessed, consistent with Australia's AI Ethics Framework (DISR, 2023).

**5. BERT model transparency**
BERT is a black-box model. Its internal attention mechanisms are not easily interpretable, raising accountability concerns in high-stakes classification contexts.

---

## Individual Contributions

---

### 👤 Nomayer Hossain | Student ID: A00176827

**Role: Data Pipeline, LSTM Model, BERT Model, Project Architecture**

**Data Pipeline**
- Designed and implemented the full data loading pipeline for the JHU Multi-Domain Sentiment Dataset, parsing pseudo-XML review files across all four product categories
- Implemented text preprocessing: lowercasing, HTML tag removal, punctuation stripping, stopword removal, and lemmatisation using NLTK
- Conducted outlier removal and balanced class verification
- Built vocabulary encoding (38,933 unique words), integer sequence mapping, and padding/truncation to fixed 200-token sequences

**LSTM Model (`model_lstm.ipynb`)**
- Designed and implemented a Bidirectional LSTM architecture in PyTorch from scratch including Embedding, BiLSTM, Dropout, and Fully Connected layers
- Implemented training loop with early stopping, gradient clipping, and Adam optimiser
- Achieved test accuracy of 76.88%
- Implemented inference function returning Positive/Negative classification with confidence score

**BERT Model (`model_bert.ipynb`)**
- Implemented BERT fine-tuning using HuggingFace Transformers on the same dataset
- Configured AdamW optimiser with linear warmup scheduler across 3 epochs on T4 GPU
- Achieved validation accuracy of 91.95%
- Saved fine-tuned model weights to Google Drive for web interface integration

**Project Architecture**
- Initialised GitHub repository, drafted README, and established folder structure

---

### 👤 Andrew Chang — Student ID: A00031568

**Role: Coordinator, Interface, Video Editor**

- Coordinate group meetings and makes group meeting appointments
- Discuss and facilitate separation of roles and task for each role
- Create presentation slides for group presentation, and iron out the format of presentation.
- Join and edit presentation videos from each member to hand in.

---

### 👤 Kelly Thaiane Costa de Araujo — Student ID: [To be updated]

**Role: [Kelly to complete]**

[Kelly — please update this section with your contribution.

---
