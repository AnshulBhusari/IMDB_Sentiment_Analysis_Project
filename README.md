# Comparative Sentiment Analysis on IMDb Movie Reviews

## 1. Title and Short Description
This project successfully implemented and benchmarked two distinct machine learning models—a **Logistic Regression (LR)** classifier using TF-IDF features as a baseline, and a **Long Short-Term Memory (LSTM) Recurrent Neural Network** leveraging Word Embeddings—for **Binary Sentiment Classification** on the IMDb 50k dataset. The analysis demonstrates the superior performance of the sequence-aware LSTM model in metrics like F1-Score and AUC-ROC, confirming the advantage of deep learning for capturing contextual dependencies in Natural Language Processing (NLP) tasks.

---

## 2. Dataset Source and Preprocessing
* **Source:** [IMDb Dataset of 17k Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Data Size:** 17,000 text reviews, perfectly balanced (8,500 positive, 8,500 negative).
* **Preprocessing Pipeline:**
    1.  **Label Encoding:** 'positive' and 'negative' labels were converted to numerical values ($1$ and $0$).
    2.  **Text Cleaning:** All text was **lowercased**, common noise like HTML tags and punctuation was removed, and standard English **stop words** were filtered out to enhance feature relevance.
    3.  **Data Split:** The dataset was stratified and partitioned into a **Training Set (80%)**, a **Validation Set (10%)** for tuning, and an **unseen Test Set (10%)** for final evaluation.

---

## 3. Methods and Model Architectures

Our strategy involved establishing a strong traditional baseline (LR) and comparing its performance against a sequential Deep Learning (DL) model (LSTM) to measure the value of context-aware feature learning.

### Model A: Logistic Regression (Baseline)
* **Feature Engineering:** Cleaned text was transformed using a **TF-IDF Vectorizer**. We used a vocabulary limited to 5000 features and included **bigrams** (n-gram range of 1 to 2) to capture simple word co-occurrence.
* **Hyperparameter Tuning:** The regularization parameter **$C$** was optimized using **GridSearchCV** with 5-fold cross-validation on the training set, optimizing for the F1-Score to achieve the best generalizable baseline performance.

### Model B: LSTM (Sequence Model) 
* **Feature Engineering:** Sequences were generated via **Tokenization** (limiting vocabulary to 10k words) and then normalized to a fixed length of 256 tokens using **Padding/Truncation**.
* **Architecture:** A Sequential model structure was implemented:
    * **Embedding Layer:** Maps integer inputs to dense, 128-dimensional vector representations.
    * **LSTM Layer (128 units):** The core layer for processing sequential data and capturing long-term dependencies.
    * **Dropout (0.3):** Applied after the Embedding and LSTM layers to prevent overfitting.
    * **Dense Output Layer:** Single neuron with a **Sigmoid** activation for binary probability output.
* **Training:** The model was compiled with the Adam optimizer and **Binary Cross-Entropy** loss. **Early Stopping** was used, monitoring the validation loss to ensure the best performing weights were saved and overfitting was mitigated.

---

## 4. Experiments/Results Summary

The final evaluation was conducted on the reserved **Test Set**. We used **F1-Score** and **AUC-ROC** as the primary metrics for a robust, threshold-independent assessment of classification quality.

### Comparative Performance Table

| Model | Feature Type | Accuracy | **F1-Score** | **AUC-ROC** | Training Time (Approx.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | TF-IDF (Bigrams) | 0.8879 | 0.8863 | 0.9528 | 0.2069 |
| **LSTM (Deep Learning)** | Word Embeddings | 0.5108 | 0.1396 | 0.5030 | 615.1619 |

### Key Findings and Analysis 
Thank you for providing the final results! This immediately highlights a crucial finding: your Logistic Regression baseline performed extremely well, but your LSTM model experienced a significant failure (near random chance).This is fantastic for an academic report because it requires a deeper analysis and justification, which will demonstrate excellent analytical skills.Here is the complete, customized analytical text for your README.md. Paste this directly into Sections 4 and 6, replacing the old placeholders.📝 Customized README.md Analysis1. Section 4: Experiments/Results Summary (Final Table)ModelFeature TypeAccuracyF1-ScoreAUC-ROCTraining Time (Approx.)Logistic Regression (LR)TF-IDF (Bigrams)0.88790.88630.95280.21 secondsLSTM (Deep Learning)Word Embeddings0.51080.13960.5030615.16 seconds (~10.25 mins)2. Section 4: Key Findings and Analysis (Text)The final evaluation revealed a sharp contrast between the models, leading to a critical finding:Exceptional Baseline Performance: The Logistic Regression model achieved an outstanding AUC-ROC of 0.9528 and an F1-Score of 0.8863. This confirms that the combination of meticulous preprocessing (stop word removal, punctuation cleaning) and robust feature engineering (TF-IDF bigrams) is highly effective for this dataset, creating a very strong, fast-to-train baseline.LSTM Model Failure: The LSTM model performed near random chance, with an Accuracy of 0.5108 and an AUC-ROC of 0.5030. This indicates a severe issue during the deep learning pipeline, likely model overfitting or convergence failure. Despite the sequential data handling, the model failed to generalize, suggesting the complexity of the neural network was too great for the limited training schedule (10 epochs with early stopping) or the initialization was poor.Trade-off: The LR model trained in a fraction of a second (0.21 seconds), while the LSTM required over 10 minutes (615.16 seconds) of GPU time to yield a near-useless result. This starkly highlights the computational risk of applying complex deep learning methods without extensive tuning.

---

## 5. Steps to Run the Code
1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    ```
2.  **Install Dependencies:** Install libraries listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute Notebooks:** Run the notebooks sequentially in a Jupyter environment (Google Colab is highly recommended for GPU support for the LSTM):
    * `01_Preprocessing_and_Model_A_LR.ipynb`
    * `02_Model_B_LSTM_Training.ipynb`

---

## 6. Conclusion
This comparative study successfully demonstrated that while a traditional machine learning approach (Logistic Regression on TF-IDF) provides a fast, strong baseline, a deep learning sequence model (LSTM) is not guaranteed to outperform without significant hyperparameter optimization.

Based on the empirical evidence, the Logistic Regression model is the clear winner for deployment. It provides a highly accurate, robust, and immediately usable classifier (AUC-ROC: 0.9528) with minimal computational cost.

The experiment concludes that the complexity and time required to optimize the LSTM model to exceed the performance of the well-engineered LR baseline were not justified in this case, making Logistic Regression the superior choice for production efficiency and reliability.
---

## 7. References
* Dataset: IMDb 50k Movie Reviews (Kaggle)
* Libraries: Scikit-learn, TensorFlow/Keras, NLTK.
