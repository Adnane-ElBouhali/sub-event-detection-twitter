# Sub-Event Detection in Twitter Streams

This repository contains the implementation of a machine learning pipeline for real-time detection of sub-events during football matches based on tweets. The project leverages state-of-the-art natural language processing techniques to identify and classify sub-events such as goals, penalties, and red cards from large-scale social media data.

## Overview

Social media platforms like Twitter generate massive amounts of real-time data during high-profile events like the FIFA World Cup. This project uses tweets to detect specific football match sub-events in real-time, demonstrating the potential of AI in extracting structured insights from unstructured data.

## Features

- **BERT-based Model**: Fine-tuned a `bert-base-uncased` model for binary classification to detect event presence (goal, penalty, red card, etc.).
- **Scalable Pipeline**: Implemented efficient data preprocessing, feature extraction, and GPU memory optimization to handle millions of tweets.
- **Imbalanced Data Handling**: Applied techniques like weighted loss functions, oversampling, and focal loss to address class imbalance.
- **Real-Time Inference**: Enabled prediction of sub-events from new, unseen tweet streams.

## Dataset

- Tweets from football matches during the 2010 and 2014 FIFA World Cups.
- Each tweet is labeled with binary classes (`1`: Sub-event present, `0`: Sub-event absent).

## Model Architecture

- **Tokenizer**: `BertTokenizer` for tokenizing and encoding tweet text.
- **Model**: Fine-tuned `BertForSequenceClassification` with a custom learning rate scheduler and gradient checkpointing.
- **Training Framework**: PyTorch and HuggingFace Transformers.

## Key Technologies

- Python, PyTorch, HuggingFace Transformers
- NVIDIA CUDA, Mixed Precision Training
- Data processing: Pandas, scikit-learn

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/sub-event-detection-twitter.git
cd sub-event-detection-twitter
```

### 2. Install Dependencies
Create a virtual environment and install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 3. Preprocess the Data
Organize the dataset into the expected format:
```bash
python preprocess.py
```

### 4. Train the Model
Train the BERT model on the preprocessed data:
```bash
python train.py
```

### 5. Evaluate the Model
Evaluate the model on the validation dataset:
```bash
python evaluate.py
```

### 6. Make Predictions
Run predictions on the evaluation dataset:
```bash
python predict.py
```

## Future Enhancements

- Extend to multi-class classification for finer sub-event detection (e.g., distinguishing goals from penalties).
- Deploy the model as a real-time API for live event monitoring.

## Contributions

Contributions and suggestions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

Replace `your_username` with your GitHub username when cloning. Add any additional details specific to your implementation.
