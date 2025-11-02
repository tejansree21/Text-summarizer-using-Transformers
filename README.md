# Text Summarizer using Transformers

## Project Overview
This project implements a **Text Summarization System** using **Transformer-based models** such as **BART** and **T5**.  
It automatically generates concise and meaningful summaries from long passages or articles using pre-trained NLP models.

---

## Features
- Abstractive Summarization using Hugging Face Transformers  
-  Handles long-form text input (news, articles, or research papers)  
-  Plug-and-play — no training required for inference  
-  Extendable for fine-tuning using datasets like CNN/DailyMail  

---

## Model Used
- **facebook/bart-large-cnn** — pretrained on CNN/DailyMail corpus for abstractive summarization  
- Alternative models:
  - `t5-small`
  - `t5-base`
  - `google/pegasus-xsum`

---

## Requirements
Install dependencies with:
bash
pip install transformers datasets torch sentencepiece

Text-Summarizer/
│
├── text_summarizer.py          # Main summarization script
├── news_article.txt            # Optional input file for custom text
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies


