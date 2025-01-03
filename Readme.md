# Clinical Text Clustering 

This project focuses on clustering clinical documents using advanced Natural Language Processing (NLP) and machine learning techniques.

## **Pipeline Workflow**

1. **Load Data**:
   - Extract clinical documents from a ZIP file into a structured format using `load_and_extract_files()`.

2. **Preprocessing**:
   - Clean and preprocess text using `TextPreprocessor`.

3. **Feature Extraction**:
   - Generate DTMs using TF-IDF.

4. **Topic Modeling**:
   - Apply NMF and LDA to identify meaningful topics.
   - Visualize topics with bar charts and top words.

5. **Dimensionality Reduction**:
   - Use UMAP to reduce feature space for clustering visualization.

6. **Clustering**:
   - Apply KMeans and MiniBatchKMeans to group documents.

7. **Visualization**:
   - Plot cluster results in 2D space to analyze document groupings.

---

### **Dependencies**
- Python 3.8+
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `umap-learn`
  
---

### **Documentation**
- documentation.ipynb
