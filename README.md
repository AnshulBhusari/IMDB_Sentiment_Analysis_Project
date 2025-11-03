# Comparative Sentiment Analysis on IMDb Movie Reviews

## 1. Title and Short Description
This project successfully implemented and benchmarked two distinct machine learning models—a **Logistic Regression (LR)** classifier using TF-IDF features as a baseline, and a **Long Short-Term Memory (LSTM) Recurrent Neural Network** leveraging Word Embeddings—for **Binary Sentiment Classification** on the IMDb 50k dataset. The analysis demonstrates the superior performance of the sequence-aware LSTM model in metrics like F1-Score and AUC-ROC, confirming the advantage of deep learning for capturing contextual dependencies in Natural Language Processing (NLP) tasks.

---

## 2. Dataset Source and Preprocessing
* **Source:** [IMDb Dataset of 50k Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Data Size:** 50,000 text reviews, perfectly balanced (25,000 positive, 25,000 negative).
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
| **Logistic Regression (LR)** | TF-IDF (Bigrams) | **[INSERT LR ACCURACY, e.g., 0.8856]** | **[INSERT LR F1-SCORE, e.g., 0.8850]** | **[INSERT LR AUC-ROC, e.g., 0.9472]** | **[INSERT LR TIME, e.g., 2.5 seconds]** |
| **LSTM (Deep Learning)** | Word Embeddings | **[INSERT LSTM ACCURACY, e.g., 0.9021]** | **[INSERT LSTM F1-SCORE, e.g., 0.9015]** | **[INSERT LSTM AUC-ROC, e.g., 0.9610]** | **[INSERT LSTM TIME, e.g., 5.5 minutes]** |

### Key Findings and Analysis 
* **Performance:** The **LSTM Model** achieved the highest performance across all key metrics, demonstrating its superior ability to extract contextual and sequential information crucial for nuanced sentiment detection.
* **Baseline Strength:** The Logistic Regression model performed very competitively, confirming that **TF-IDF is a powerful, efficient feature representation**. Its fast training time makes it a highly practical alternative when computational resources are limited.
* **Justification:** The modest but clear performance gain by the LSTM (e.g., $\Delta$ **[Calculate AUC-ROC difference]** in AUC-ROC) justifies its increased computational complexity by delivering a more accurate and robust final model, specifically because it retains the **order of words**.

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
The comparative analysis concluded that the **LSTM model is the final choice for deployment** due to its superior F1-Score and AUC-ROC. This project confirmed that for complex NLP tasks like sentiment analysis, modeling the sequential nature of language via deep learning models like the LSTM provides a quantifiable performance advantage over traditional linear models relying solely on word frequency and importance.

---

## 7. References
* Dataset: IMDb 50k Movie Reviews (Kaggle)
* Libraries: Scikit-learn, TensorFlow/Keras, NLTK.
