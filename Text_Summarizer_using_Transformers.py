# Text Summarization using Transformers (BART / T5)
# You can choose either BART or T5 for summarization.

from transformers import pipeline
# 1️⃣ Load Summarization Pipeline
# ----------------------------
# Model options: "facebook/bart-large-cnn", "t5-small", "t5-base", etc.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Input Text
text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
that can perform tasks that typically require human intelligence. 
These tasks include reasoning, learning, problem-solving, perception, and language understanding. 
AI is impacting every sector — from healthcare and education to finance and space exploration.
"""

# Generate Summary
summary = summarizer(
    text,
    max_length=60,     # controls the summary length
    min_length=25,     # minimum length of summary
    do_sample=False    # deterministic (greedy decoding)
)

# Print the Output
print("\nOriginal Text:\n", text)
print("\nGenerated Summary:\n", summary[0]['summary_text'])

# Summarize Custom Articles
article = open("news_article.txt", "r", encoding="utf-8").read()
summary = summarizer(article, max_length=120, min_length=50, do_sample=False)
print(summary[0]['summary_text'])

# Dataset (Optional for Training)
# you can use the CNN/DailyMail dataset if you would like to fine tune your own model 
# This dataset contains news articles paired with summaries, prefect with fine-tuning
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
print(dataset["train"][0])

