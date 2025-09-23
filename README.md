# 🛍️ Multimodal Product Attribute Extraction & Normalization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![HuggingFace](https://img.shields.io/badge/Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-green)



## 📌 Project Overview
This project builds a multimodal deep learning pipeline to **extract and normalize product attributes** (brand, color, size, material, category) from both **product text (title + description)** and **images**.

E-commerce catalogs like Amazon rely on clean attributes for:
- Better search & filtering
- Improved recommendations
- Accurate ads targeting
- Catalog standardization at scale

## 🎯 Goals
- Extract attributes using **text + image models**
- Normalize noisy values into a canonical schema
- Compare **text-only, image-only, and multimodal fusion** performance
- Deliver a working demo where users upload product data and see extracted attributes

## ⚙️ Architecture
1. **Text Encoder**: BERT/DistilBERT → extract features from titles + descriptions
2. **Image Encoder**: ViT/CLIP → extract visual features
3. **Fusion Model**: Concatenate embeddings → predict attributes
4. **Normalization**: Fuzzy matching + embeddings to map outputs to canonical schema

## 📊 Datasets
- [Amazon Product Metadata (SNAP)](https://snap.stanford.edu/data/)
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [Stanford Online Products](http://cvgl.stanford.edu/projects/lifted_struct/)

## 🔧 Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Torchvision, CLIP (OpenAI), Scikit-learn
- FuzzyWuzzy / RapidFuzz (string normalization)
- Streamlit (demo)

## 🚀 How to Run
```bash
# clone repo
git clone https://github.com/your-username/multimodal-attribute-extraction.git
cd multimodal-attribute-extraction

# setup environment
pip install -r requirements.txt

# run demo
streamlit run demo/app.py

```

Results (Sample)
Model	F1 Score	Normalization Accuracy
Text-only (BERT)	0.72	81%
Image-only (ViT)	0.65	76%
Multimodal Fusion	0.82	89%
