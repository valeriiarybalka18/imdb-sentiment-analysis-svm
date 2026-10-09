# IMDb Sentiment Analysis using a Linear SVM in TensorFlow/Keras

A binary sentiment analysis project built from scratch using the IMDb dataset. This project implements a linear Support Vector Machine (SVM) architecture using a single-layer Keras model combined with TF-IDF vectorization.

## 🚀 Key Features & Methodology
- **Data Representation**: Text data is transformed using **TF-IDF Vectorization** (limited to 10,000 features) to convert reviews into numerical sparse matrices.
- **Model Architecture**: A custom linear classifier implemented via a single Keras `Dense(1)` layer.
- **Loss Function**: Trained using **Hinge Loss** coupled with the Adam optimizer to replicate a linear SVM objective.
- **Evaluation**: Custom evaluation based on sign thresholding of continuous predictions, bypassing standard binary classification threshold issues with `[-1, 1]` labels.

## 📊 Results
- **Test Accuracy**: Achieved **~88% accuracy** on the IMDb test dataset.
- **Feature Importance Analysis**: Extracted the model's learned weights directly from the dense layer to discover top predictive tokens.

### Top Learned Features
- **Top Positive Words**: `loved`, `wonderful`, `best`, `perfect`, `amazing`, `excellent`, `great`
- **Top Negative Words**: `worst`, `awful`, `poor`, `bad`, `boring`, `waste`, `terrible`

## 🛠️ Tech Stack
- **Python** (Jupyter Notebook)
- **TensorFlow / Keras**
- **Scikit-learn** (TF-IDF Vectorizer)
- **NumPy**
