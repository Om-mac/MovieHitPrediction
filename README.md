# 🎬 Movie Hit Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Ensemble-green.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> 🔮 Predict whether a movie will be a **HIT** or **FLOP** using Machine Learning, NLP, and Streamlit

An end-to-end machine learning pipeline that predicts the commercial success of movies using metadata, cast/director success metrics, production company reputation, sentiment analysis, and a stacked ensemble model combining Random Forest and XGBoost.

---

## 🌟 Features

### ✨ Machine Learning Pipeline
- **Data Processing**: Automated cleaning of TMDB dataset (movies + credits)
- **Advanced Feature Engineering**:
  - Cast success score (average revenue of cast members)
  - Director success score
  - Production company reputation metrics
  - Multi-hot encoded genres
  - Sentiment scoring (BERT-ready architecture)
  - Hype score (popularity + vote statistics)
- **Label Generation**: Intelligent Hit/Flop classification using ROI
- **Ensemble Models**:
  - Random Forest Classifier
  - XGBoost Classifier
  - Stacked Meta-Learner (Logistic Regression)

### 🎯 Interactive Web Application
Built with Streamlit for real-time predictions:

**Input Parameters**:
- Budget
- Popularity score
- Runtime
- Vote statistics
- Cast names
- Director
- Production company

**Output Analytics**:
- Hit probability score
- Ensemble prediction result
- Base model probabilities
- Complete feature vector
- Feature importance visualization

---

## 📂 Project Structure

```
MovieHitPrediction/
│
├── data/
│   ├── tmdb_5000_movies.csv          # Main movies dataset
│   ├── tmdb_5000_credits.csv         # Cast and crew data
│   └── imdb_reviews/                 # (Optional) Review data
│
├── artifacts/
│   ├── xgb_model.joblib              # Trained XGBoost model
│   ├── rf_model.joblib               # Trained Random Forest model
│   ├── meta_clf.joblib               # Meta-classifier model
│   └── movies_featured.csv           # Engineered features dataset
│
├── MasterMovieHit_Pipeline.py        # Complete training pipeline
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Random Forest | 0.8139 | 0.9259 |
| XGBoost | 0.8604 | 0.9182 |
| **Stacked Ensemble** | **0.8837** | **0.9313** |

> 🏆 The stacked ensemble model achieves the best performance and is deployed in the Streamlit application.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Om-mac/MovieHitPrediction.git
cd MovieHitPrediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Dependencies**:
```bash
pip install pandas numpy scikit-learn xgboost streamlit transformers torch rapidfuzz joblib shap matplotlib seaborn
```

### 3️⃣ Download Dataset
Place the following datasets in the `/data` folder:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

**Dataset Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 🚀 Usage

### Train the Model
Run the complete training pipeline:
```bash
python MasterMovieHit_Pipeline.py
```

This will:
- Process and clean the datasets
- Engineer advanced features
- Train all models
- Save artifacts to `./artifacts/`

### Launch the Web Application
```bash
streamlit run app.py
```

The app will be available at: `http://localhost:8501`

---

## 🧠 How It Works

### 🔍 Feature Engineering Pipeline

Each movie is represented by a comprehensive feature vector:

1. **Basic Features**:
   - Budget
   - Popularity score
   - Runtime
   - TMDB rating
   - Vote count

2. **Success Metrics**:
   - Cast success score (average revenue of cast members)
   - Director success score
   - Production company success score

3. **Content Features**:
   - Genre embeddings (multi-hot encoding)
   - Sentiment score (BERT-based or rating proxy)
   - Hype score (normalized popularity + votes)

### 🤖 Prediction Pipeline

```
Input Features
    ↓
┌─────────────────┐
│ Random Forest   │ → Probability₁
└─────────────────┘
    ↓
┌─────────────────┐
│   XGBoost       │ → Probability₂
└─────────────────┘
    ↓
┌─────────────────┐
│ Meta-Learner    │ → Final Prediction
└─────────────────┘
```

### 🎯 Output

- **Hit Probability**: Confidence score (0.0 - 1.0)
- **Prediction Label**: HIT or FLOP
- **Feature Contributions**: SHAP values for interpretability

---

## 🧪 Example Prediction

### Input
```python
{
    "budget": 50000000,
    "cast": ["Robert Downey Jr", "Chris Evans"],
    "director": "James Cameron",
    "popularity": 30.5,
    "vote_average": 7.5,
    "runtime": 142,
    "production_company": "Marvel Studios"
}
```

### Output
```
✅ Prediction: HIT
🎯 Hit Probability: 0.97 (97%)
📊 Confidence: Very High

Feature Contributions:
  - Cast Success Score: +0.35
  - Director Reputation: +0.28
  - Production Company: +0.18
  - Budget-to-Hype Ratio: +0.16
```

---

## 📈 Future Enhancements

- [ ] 🎥 YouTube trailer sentiment analysis + view count integration
- [ ] 🐦 Twitter/X hype scoring and social media analytics
- [ ] 🧠 Fine-tuned BERT sentiment model (IMDB 50K dataset)
- [ ] 💵 Inflation-adjusted ROI calculations
- [ ] 🎞️ Bollywood and international dataset integration
- [ ] 📊 Real-time model monitoring dashboard
- [ ] 🌐 REST API deployment
- [ ] 📱 Mobile application

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🧑‍💻 Author

**Om Tapdiya**

- 🐙 GitHub: [@Om-mac](https://github.com/Om-mac)
- 💼 LinkedIn: [linkedin.com/in/omtapdiya](https://www.linkedin.com/in/omtapdiya)
- 📧 Email: Omtapdiya75@gmail.com

---

## 🙏 Acknowledgments

- TMDB for providing the comprehensive movie dataset
- Scikit-learn and XGBoost communities for excellent ML libraries
- Streamlit for the intuitive web framework
- The open-source community for continuous inspiration

---

## 📧 Contact

For questions or suggestions, please reach out:
- Email: Omtapdiya75@gmail.com
- Issues: [GitHub Issues](https://github.com/Om-mac/MovieHitPrediction/issues)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ and ☕ by Om Tapdiya

</div>