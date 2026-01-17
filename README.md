# Hinglish Sentiment Analysis using DistilBERT  
A Real-Time Sentiment Classification System for Code-Mixed Hindi–English Reviews

This repository contains the implementation and research work for a DistilBERT-based sentiment analysis model designed to classify Hinglish (Hindi-English code-mixed) customer reviews. The model is optimized for real-time performance, achieving fast inference suitable for e-commerce platforms, chatbots, and live feedback monitoring systems.

---

## 🚀 Project Overview

India’s e-commerce platforms receive thousands of reviews written in a mix of Hindi and English. Traditional NLP models struggle with such code-mixed text. This project fine-tunes DistilBERT (multilingual) to classify sentiments as:

- Positive
- Negative
- (Optional) Neutral

The system is optimized for low latency and performs inference in ~65 ms, making it suitable for real-time applications.

---

## 🧠 Key Features

- ✔ Code-mixed Hinglish dataset preprocessing  
- ✔ DistilBERT multilingual fine-tuning  
- ✔ Evaluation metrics (accuracy, F1-score, precision, recall)  
- ✔ Real-time inference speed measurement  
- ✔ Streamlit demo UI (optional integration)  
- ✔ Research-paper ready documentation  

---

## 📂 Project Structure

```



## 📊 Model Performance

| Metric     | Score |
|-----------|-------|
| Accuracy  | 62.21% |
| Precision | 0.6165 |
| Recall    | 0.57 |
| F1-score  | 0.5763 |

Despite noisy real-world text, DistilBERT shows strong performance for code-mixed inputs.

---

## 🚀 Latency (Speed) Performance

- **Average inference time:** ~65.87 ms
- Suitable for:
  - Live chatbots  
  - Customer support dashboards  
  - Automated review monitoring  
  - Real-time sentiment tracking  

---


## 🤖 Real-Time Prediction

```python
python src/predict.py
```

Example:

```python
text = "Product accha hai but delivery late thi"
print(predict(text))
```


## 📘 Research Paper

The full research paper is included:

📄 **paper/second_research_paper.pdf**

Based on:

* DistilBERT multilingual model
* 2,766 Hinglish reviews
* Transformer-based transfer learning
* Real-time performance benchmarks

---

## 📌 Future Improvements

* Improve class imbalance with oversampling
* Add Neutral class training
* Expand dataset to 10k+ samples
* Experiment with XLNet, IndicBERT, MuRIL

---

## 🤝 Contributing

Pull requests are welcome!
Please ensure that any feature additions include documentation and tests.

---


## 👤 Author

**Rudra Akhauri**
Department of Computer Science
IMPACT College
India

```

