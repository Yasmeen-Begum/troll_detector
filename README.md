
# Troll Detector

Detect trolls vs. genuine users in online text using NLP and machine learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Sample Input/Output](#sample-inputoutput)
- [Model Explainability](#model-explainability)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Troll Detector** is a Python-based system for classifying online users as **trolls** or **genuine users** based on their text behavior and toxicity levels. It aggregates user messages, extracts linguistic and behavioral features, scores toxicity, and uses a supervised machine learning model to detect trolls. The system is modular, explainable, and easily extensible.

## Features

- Aggregates multiple messages per user for robust analysis
- Standard NLP preprocessing (tokenization, stopword removal, lemmatization)
- Toxicity detection using keyword-based or model-based scoring
- Behavioral and sentiment feature extraction
- Supervised classification (Random Forest)
- Model explainability (feature importances)
- Evaluation with accuracy, precision, recall, F1-score
- Easily extensible for advanced toxicity models or dashboards

## Project Structure

```
troll_detector/
├── data/
│   └── Dataset for Detection of Cyber-Trolls.json
├── models/
│   └── troll_detector.joblib
├── src/
│   ├── preprocessing.py
│   ├── toxicity.py
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   └── evaluate.py
├── main.py
├── requirements.txt
├── README.md
└── Troll_detector_report.pdf
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yasmeen-Begum/troll_detector.git
   cd troll_detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Place your `Dataset for Detection of Cyber-Trolls.json` in the `data/` directory.

## Usage

1. **Run the pipeline:**
   ```bash
   python main.py
   ```

2. **What happens:**
   - Loads and preprocesses the dataset.
   - Aggregates user messages and extracts features.
   - Trains and evaluates a Random Forest classifier.
   - Prints evaluation metrics and feature importances.

## Sample Input/Output

**Input:**  
A JSON Lines file with:
- `content`: user message text
- `annotation`: dictionary with `label` field (e.g., `{'label': ['1']}`)

**Output:**  
- Console: Evaluation metrics (accuracy, precision, recall, F1), and feature importances.
- File: Trained model saved as `models/troll_detector.joblib`.

## Model Explainability

After training, the script displays feature importances, showing which behavioral and linguistic features most influenced the troll/genuine classification. This helps users and moderators understand the model's decisions.

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.

## License

This project is licensed under the MIT License.


