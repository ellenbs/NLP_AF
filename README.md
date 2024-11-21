# Classifying Sentiments on the TweetEval Dataset

### Natural Language Processing (NLP) Project | Deep Learning Researcher

Student: Ellen Shen

This repository contains the code and resources for fine-tuning the **RoBERTa-Base** model to classify sentiments on the **TweetEval** dataset. The project focuses on the task of sentiment analysis with three predefined classes: **Negative**, **Neutral**, and **Positive**, and compares the results with the benchmarks reported in the TweetEval paper.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Research on Previous Work](#research-on-previous-work)
3. [Dataset](#dataset)
4. [Model and Methods](#model-and-methods)
5. [Results](#results)
6. [References](#references)

---

## Introduction
Sentiment analysis is a key NLP task that involves classifying text into sentiment categories. This project leverages the **TweetEval** dataset, a benchmark designed for tweet-specific classification tasks, and fine-tunes **RoBERTa-Base** to achieve competitive results with a **Macro-Averaged Recall** of 0.71, close to the benchmarks in the TweetEval paper.

---

## Research on Previous Work

Sentiment analysis has been extensively studied as a key task in Natural Language Processing (NLP). This project builds on prior work by leveraging transformer-based architectures for tweet-specific sentiment classification. Below, we summarize relevant research:

1. TweetEval Benchmark

    The TweetEval benchmark, introduced by Barbieri et al. [1], provides a unified framework for evaluating tweet classification tasks, including sentiment analysis. The authors highlight challenges in handling noisy and informal tweet data and propose Macro-Averaged Recall as a robust metric for imbalanced datasets. Their work demonstrates the effectiveness of RoBERTa-Twitter, a variant of RoBERTa fine-tuned on Twitter-specific data, achieving a Macro-Averaged Recall of 72.9.

2. Transformer Models for Sentiment Analysis

    Liu et al. [2] introduced RoBERTa, an optimized variant of BERT, which outperforms previous transformer architectures on several NLP benchmarks. Its robust pretraining strategies make it a suitable candidate for fine-tuning on specialized tasks such as sentiment analysis.

3. Challenges in Sentiment Analysis on Tweets

    Research by Zhang et al. [3] focuses on the challenges of tweet sentiment analysis, including handling:

    Abbreviations and emojis.
    Domain-specific terminology.
    Imbalanced data distributions.
    Their findings suggest that fine-tuning large pre-trained models with data augmentation or domain-specific pretraining can significantly improve performance.

4. Comparison with Traditional Approaches

    Traditional methods for sentiment analysis relied on feature engineering and classical machine learning algorithms (e.g., Logistic Regression, SVM). However, studies like Socher et al. [4] show that neural network-based methods, especially transformers, outperform classical approaches by capturing semantic nuances.

### Inspiration for This Project

Leveraging RoBERTa as a baseline transformer model due to its strong performance across diverse NLP tasks [2].
Using the TweetEval dataset and its evaluation metrics [1] to standardize performance comparison.
Addressing class imbalance through dataset balancing techniques inspired by [3].
Evaluating results against the Macro-Averaged Recall benchmark of 72.9 reported by Barbieri et al. [1].

---

## Dataset
The **TweetEval** dataset includes three sentiment classes:
- **Negative**: Tweets with negative sentiment.
- **Neutral**: Tweets with neutral sentiment.
- **Positive**: Tweets with positive sentiment.

The dataset is inherently noisy and imbalanced, making it challenging to achieve high performance across all classes.

For more details on the dataset, refer to the [TweetEval repository](https://github.com/cardiffnlp/tweeteval).

---

## Model and Methods
The project fine-tunes **RoBERTa-Base**, a transformer-based model, using the Hugging Face library. Key steps include:

1. **Preprocessing**:
   - Tokenization using `RobertaTokenizer` with truncation and padding to a maximum length of 128 tokens.
2. **Training**:
   - Fine-tuned using a balanced subset of the training data (500 samples per class).
   - Hyperparameters:
     - Learning Rate: `2e-5`
     - Batch Size: `16`
     - Epochs: `3`
3. **Evaluation**:
   - Metrics: Precision, Recall, F1-Score, and Macro-Averaged Recall.
   - Test set includes 12,284 examples with the original class distribution.

---

## Results
### Performance Metrics
| **Metric**           | **Value** |
|-----------------------|-----------|
| **Macro-Averaged Recall** | 0.71      |
| **Accuracy**          | 66%       |

### Detailed Metrics by Class
| **Class**     | **Precision** | **Recall** | **F1-Score** |
|---------------|---------------|------------|--------------|
| Negative      | 0.61          | 0.86       | 0.72         |
| Neutral       | 0.78          | 0.47       | 0.59         |
| Positive      | 0.60          | 0.78       | 0.68         |

These results demonstrate the model's competitive performance compared to the **TweetEval paper**, which reports a Macro-Averaged Recall of 72.9 for RoBERTa-Twitter.

---

## References

1. Barbieri, F., et al. "TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification." arXiv preprint arXiv:2010.12421, 2020.
2. Liu, Y., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv preprint arXiv:1907.11692, 2019.
3. Zhang, Z., et al. "Sentiment Analysis of Twitter Data Using Machine Learning Approaches and Semantic Analysis." Journal of Big Data, 2018. Link
4. Socher, R., et al. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank." EMNLP, 2013.
