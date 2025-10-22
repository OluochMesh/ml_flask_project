# ml_flask_project# Machine Learning Flask Portfolio

A comprehensive machine learning web application demonstrating Classical ML, Deep Learning, and Natural Language Processing capabilities using Flask, scikit-learn, TensorFlow, and spaCy.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project implements three comprehensive machine learning tasks:

1. **Task 1: Iris Classification** - Classical ML with Decision Trees
2. **Task 2: MNIST Digit Recognition** - Deep Learning with CNNs
3. **Task 3: NLP Analysis** - Named Entity Recognition & Sentiment Analysis

## 🚀 Features

### Task 1: Iris Species Classification
- ✅ Decision Tree classifier with hyperparameter tuning
- ✅ Data preprocessing with missing value handling
- ✅ Comprehensive evaluation metrics (Accuracy, Precision, Recall)
- ✅ Interactive prediction interface
- ✅ Confusion matrix and feature importance visualizations

### Task 2: MNIST Handwritten Digit Recognition
- ✅ Convolutional Neural Network (CNN) architecture
- ✅ >95% test accuracy achievement
- ✅ Real-time digit prediction from uploaded images
- ✅ Model training with callbacks and early stopping
- ✅ Prediction confidence scores

### Task 3: NLP Analysis on Amazon Reviews
- ✅ Named Entity Recognition (NER) with spaCy
- ✅ Extract product names and brand mentions
- ✅ Rule-based sentiment analysis
- ✅ Batch review processing
- ✅ Sentiment keyword extraction

## 📁 Project Structure

```
ml_flask_project/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── routes/                  # Route blueprints
│   │   ├── main.py             # Main routes
│   │   ├── task1_routes.py     # Iris classification
│   │   ├── task2_routes.py     # MNIST recognition
│   │   └── task3_routes.py     # NLP analysis
│   ├── models/                  # ML models
│   │   ├── task1_iris/         # Iris classifier
│   │   ├── task2_mnist/        # MNIST CNN
│   │   └── task3_nlp/          # NLP analyzers
│   ├── templates/              # HTML templates
│   └── static/                 # CSS, JS, images
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed data
├── trained_models/             # Saved models
├── results/                    # Metrics & visualizations
├── scripts/                    # Utility scripts
├── config.py                   # Configuration
├── run.py                      # Application entry point
└── requirements.txt            # Dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ml_flask_project
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Model

```bash
python3 -m spacy download en_core_web_sm
```

### Step 5: Prepare Datasets

#### For Task 1 (Iris):
Place your `iris.csv` file in `data/raw/`

#### For Task 2 (MNIST):
Will be downloaded automatically by TensorFlow

#### For Task 3 (Amazon Reviews):
```bash
# Place your Amazon review files in data/raw/
# Then run:
python3 scripts/prepare_amazon_data.py
```

### Step 6: Train Initial Models

```bash
# Train Iris model
python3 scripts/train_iris_model.py

# Train MNIST model
python3 scripts/train_mnist_model.py

# Test NLP components
python3 scripts/test_nlp.py
```

## 🎮 Running the Application

```bash
python3 run.py
```

The application will be available at: **http://localhost:5000**

## 📊 Usage

### Task 1: Iris Classification

1. Navigate to **Task 1** from the home page
2. Choose from three options:
   - **Predict**: Enter iris measurements for prediction
   - **Train Model**: Train with custom parameters
   - **Dataset Info**: View dataset statistics

**Example Prediction:**
```
Sepal Length: 5.1 cm
Sepal Width: 3.5 cm
Petal Length: 1.4 cm
Petal Width: 0.2 cm
→ Result: Iris Setosa (98% confidence)
```

### Task 2: MNIST Recognition

1. Navigate to **Task 2**
2. Upload a handwritten digit image (28x28 pixels recommended)
3. View prediction with confidence scores
4. See top 3 predictions with probabilities

**Supported formats:** PNG, JPG, JPEG

### Task 3: NLP Analysis

1. Navigate to **Task 3**
2. Three analysis modes:
   - **Single Review**: Analyze one review
   - **Batch Analysis**: Analyze multiple reviews
   - **Sample Reviews**: Load pre-loaded examples

**Features:**
- Sentiment classification (Positive/Negative/Neutral)
- Product and brand entity extraction
- Sentiment keywords identification
- Confidence scores

## 🧪 Testing

### Run All Tests

```bash
# Test Task 1
python3 scripts/train_iris_model.py

# Test Task 2
python3 scripts/train_mnist_model.py

# Test Task 3
python3 scripts/test_nlp.py
```

### Expected Results

**Task 1:**
- Training Accuracy: ~95%+
- Test Accuracy: ~93%+

**Task 2:**
- Training Accuracy: ~99%+
- Test Accuracy: >95% (Target achieved!)

**Task 3:**
- NER: Extracts products and brands
- Sentiment: Classifies with confidence scores

## 📈 Performance Metrics

| Task | Model | Accuracy | Parameters |
|------|-------|----------|------------|
| Task 1 | Decision Tree | 93%+ | ~1K |
| Task 2 | CNN | 99%+ | ~540K |
| Task 3 | spaCy + Rules | N/A | Lexicon-based |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model paths
IRIS_MODEL_PATH = 'trained_models/iris_decision_tree.pkl'
MNIST_MODEL_PATH = 'trained_models/mnist_cnn.h5'

# Data paths
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

# Training parameters
MNIST_EPOCHS = 10
MNIST_BATCH_SIZE = 128
```

## 📚 Technologies Used

### Backend
- **Flask 3.0.0** - Web framework
- **Python 3.10+** - Programming language

### Machine Learning
- **scikit-learn 1.3.2** - Classical ML algorithms
- **TensorFlow 2.15.0** - Deep learning framework
- **Keras 2.15.0** - Neural network API

### NLP
- **spaCy 3.7.2** - Named Entity Recognition
- **TextBlob 0.17.1** - Sentiment analysis

### Data Processing
- **pandas 2.1.4** - Data manipulation
- **numpy 1.26.2** - Numerical computing

### Visualization
- **matplotlib 3.8.2** - Plotting library
- **seaborn 0.13.0** - Statistical visualizations

### Frontend
- **Bootstrap 5.3.2** - UI framework
- **Bootstrap Icons** - Icon library
- **JavaScript** - Interactivity

## 🚧 Troubleshooting

### Model Not Found Error
```bash
# Train the model first
python3 scripts/train_iris_model.py
python3 scripts/train_mnist_model.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### spaCy Model Not Found
```bash
python3 -m spacy download en_core_web_sm
```

### Port Already in Use
```bash
# Change port in run.py
app.run(host='0.0.0.0', port=5001)
```

## 📝 API Endpoints

### Task 1 Endpoints
- `GET /task1/` - Main page
- `POST /task1/train` - Train model
- `POST /task1/predict` - Make prediction
- `GET /task1/dataset-info` - Dataset statistics

### Task 2 Endpoints
- `GET /task2/` - Main page
- `POST /task2/train` - Train CNN model
- `POST /task2/predict` - Predict digit
- `GET /task2/model-info` - Model information

### Task 3 Endpoints
- `GET /task3/` - Main page
- `POST /task3/analyze` - Analyze single review
- `POST /task3/analyze-batch` - Analyze multiple reviews
- `GET /task3/load-sample-reviews` - Load sample data

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Iris Dataset: UCI Machine Learning Repository
- MNIST Dataset: Yann LeCun's website
- Amazon Reviews: Amazon Customer Reviews Dataset
- spaCy: Explosion AI
- Flask: Pallets Projects

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: support@example.com

## 🗺️ Roadmap

- [ ] Add user authentication
- [ ] Implement model versioning
- [ ] Add more ML algorithms
- [ ] Deploy to cloud (AWS/Heroku)
- [ ] Add Docker support
- [ ] Implement REST API
- [ ] Add unit tests
- [ ] Create mobile-responsive design

---

**Built with ❤️ using Flask, TensorFlow, and spaCy**