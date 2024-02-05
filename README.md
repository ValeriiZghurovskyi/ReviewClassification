# DS part

## Overview

This project is dedicated to binary sentiment classification of movie reviews, aiming to categorize each review as either positive or negative. Utilizing a dataset of 50,000 reviews, we applied various data science and machine learning techniques to achieve this goal. This report outlines the key steps of the project, including exploratory data analysis, feature engineering, model selection, performance evaluation, and potential business applications.

## Conclusions from Exploratory Data Analysis (EDA)

- **Balanced Dataset:** The dataset comprises an equal number of positive and negative reviews, which is crucial for maintaining unbiased model training.
- **Length Analysis:** Review lengths varied significantly, with most reviews having a moderate length, suggesting that extensive text trimming was unnecessary.
- **Word Frequency Analysis:** Distinct patterns in frequently used words were observed between positive and negative reviews, indicating strong sentiment-driven language usage.

## Description of Feature Engineering

- **Text Preprocessing:** Included cleaning steps such as removing URLs, non-alphabetic characters, lowercasing, and elimination of stop words to focus on relevant text.
- **Tokenization:** Involved breaking down each review into individual words for further analysis.
- **Stemming and Lemmatization Comparison:** After evaluating both, stemming was chosen for its efficiency in reducing words to their base forms, which was more suitable for our analysis.
- **Vectorization with TF-IDF and Stemming:** We chose the TF-IDF Vectorizer combined with stemming, as it effectively captured the importance of words in the documents, providing a balance between word frequency and its relevance.

## Reasonings on Model Selection

- **Model Exploration:** Tested various models including Logistic Regression, SVM, Random Forest, and Naive Bayes, evaluating them based on accuracy, precision, recall, and F1 score.
- **Optimal Model Choice - LinearSVC with TF-IDF + Stemming:** LinearSVC was selected as the optimal model due to its superior performance in metrics, especially when combined with TF-IDF vectorization and stemming. This combination proved effective in handling high-dimensional data and offered a balance between accuracy and computational efficiency.

## Overall Performance Evaluation

The chosen LinearSVC model, in conjunction with TF-IDF vectorization and stemming, exhibited excellent performance:

- **Accuracy:** 88.23%, indicating a high success rate in classification.
- **Recall:** 89.07%, showing the model's effectiveness in correctly identifying positive reviews.
- **Precision:** 87.67%, reflecting the accuracy of positive predictions.
- **F1 Score:** 88.36%, demonstrating a balanced performance between precision and recall.

## Potential Business Applications and Value for Business

- **Enhanced Customer Insight:** Automated sentiment analysis of customer reviews for better understanding of customer satisfaction and preferences.
- **Product Review Analysis:** Efficient processing and categorization of product reviews to inform product development and marketing strategies.
- **Market Research:** Utilizing sentiment analysis for social media and market trend analysis, providing insights into public opinion and consumer behavior.
- **Content Moderation:** Automated moderation of user-generated content on platforms by identifying negative sentiments.

## Conclusion

This sentiment classification project successfully demonstrates the application of advanced data science techniques, particularly the effective use of TF-IDF with stemming and the LinearSVC model, to derive meaningful insights from text data. The project's outcomes hold significant potential for businesses in customer experience enhancement and market analysis.

# ML part

## Repo structure

/ReviewClassification/

|--data/ #This folder is ignored

|     |--raw/

|     |--processed/

|--notebooks/

|--src/

|     |--train/

|     |     |--train.py

|     |     |--Dockerfile

|     |--inference/

|     |     |--inference.py

|     |     |--Dockerfile

|     |--data_load.py

|     |--data_preparation.py

|--outputs/ #This folder is ignored

|     |--models/ #This folder stores trained models

|     |     |--model_1.pkl

|     |--predictions/ #This folder stores model predictions and their metrics

|     |     |--predictions.csv

|     |     |--metrics.txt

|     |-- vectors/ #This folder stores the vectorizer for the same data preparation

|--README.MD

|--requirements.txt

|--settings.json


## How to buid the project

### Cloning the repository

First step. Clone my repository from github, like an example:

```bash
git clone https://github.com/ValeriiZghurovskyi/ReviewClassification
```

### Data load

Go to `ReviewClassification` folder:

```bash
cd ReviewClassification
```

If you have the appropriate files for model training and inference, you can create a `data` folder:

```bash
mkdir Data
```

and `raw` folder within it:

```bash
cd Data
```

```bash
mkdir Data
```

and move your files there.

If you do not have the appropriate files for training and inference, you can upload them using the `data_load.py` script:

```bash
python3 src/data_load.py
```

This script loads the corresponding files into the `data/raw` directory using the link from the `settings.json` file. It names them `train.csv` and `inference.csv` respectively.

### Data preparation

To prepare the data, run the `data_preparation.py` script.

```bash
python3 src/data_preparation.py
```

In the `settings.json` file, you can specify which files should be prepared by setting True or False, and you also need to specify the name of the files to be prepared. By default, these are the `train.csv` and `inference.csv` files that are loaded using the previous script.

### Training

In the `settings.json` file you can customize the training parameters. You can choose the name of the model, the data file it will use for training, test_size, and the hyperparameter C

1. To train the model using Docker:

- Build the training Docker image:

```bash
docker build -f src/train/Dockerfile -t training-image .
```

- Run the container to train the model:

```bash
docker run -v ${PWD}/outputs:/app/outputs training-image
```

This command will start the process of training the model inside the container.

After executing the container, the corresponding folders and files of the model and metrics will appear on your local machine.

2. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 src/train/train.py
```

### Inference

Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `src/inference/inference.py`.

In the `settings.json` file you can customize the inference parameters. You can choose the name of the model and the data file it will use for inference.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -t inference-image -f src/inference/Dockerfile .

```

- Run the inference Docker container:
```bash
docker run -v ${PWD}/outputs:/app/outputs inference-image
```
The inference results will automatically be stored in folder on your local machine.

2. Alternatively, you can also run the inference script locally:

```bash
python src/inference/inference.py
```

