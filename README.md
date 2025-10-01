# 🛍️ Multimodal Product Attribute Extraction & Normalization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![HuggingFace](https://img.shields.io/badge/Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-green)


🚧 **Project Status: Work in Progress** 🚧  
This project is actively being developed. Current focus: training text models and preparing multimodal fusion. Some components (image pipeline, full demo, evaluation) are still under construction.  

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



✅ Current Progress

 Text baseline model (BERT / DistilBERT)

 Training loop + evaluation

 Image baseline (CLIP / ViT)

 Multimodal fusion model

 Attribute normalization module

 Streamlit demo integration

📊 Current Results (Text-only Baseline)

F1 Score: 0.72

Normalization Accuracy: 81%

🔮 Next Steps

Train & evaluate image-only baseline

Implement multimodal fusion

Add normalization layer for clean attributes

Build interactive Streamlit demo



🤝 Contribution

This is a solo project for now, but ideas and suggestions are welcome!
