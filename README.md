# Fake News Detection in English and Hindi

This project compares traditional machine learning and transformer models for classifying English and Hindi news articles as fake or real. I developed it as part of my MSc Business Analytics dissertation at Aston University.

## Approach

- Cleaned and explored a combined English and Hindi news dataset, removing duplicate records and identical texts with conflicting labels.
- Created separate stratified train and test splits for each language.
- Compared Logistic Regression, linear Support Vector Machine (SVM) and Random Forest using TF-IDF features fitted on the training data.
- Fine-tuned XLM-RoBERTa and evaluated it on the same held-out test sets used for the traditional models.
- Used LIME and SHAP to examine which words and tokens influenced selected SVM and XLM-RoBERTa predictions.

## Results

| Model | English macro F1 | Hindi macro F1 |
|---|---:|---:|
| Logistic Regression | 0.9870 | 0.9920 |
| Linear SVM | 0.9908 | 0.9937 |
| Random Forest | 0.9777 | 0.9884 |
| XLM-RoBERTa | **0.9995** | **0.9947** |

XLM-RoBERTa achieved the highest macro F1 in both languages. Linear SVM was the strongest traditional model and came within 0.0010 macro F1 of XLM-RoBERTa on the Hindi test set. These scores describe performance on this dataset's held-out splits. Performance on articles from unseen publishers or time periods remains unverified.

## How to run

1. Open [Fake_News_Detection.ipynb](Fake_News_Detection.ipynb) in Google Colab.
2. Make `combined_news_raw.csv` available to Colab and update the path in the `pd.read_csv()` cell.
3. Update `BASE_DIR` to a writable folder for saved models and results.
4. Run the notebook cells in order. Use a GPU runtime for XLM-RoBERTa fine-tuning.

## Tools

Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, LIME and SHAP.
