import os
import pickle
import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt')
nltk.download('stopwords')

def load_settings(settings_path='settings.json'):
    try:
        with open(settings_path, 'r') as file:
            settings = json.load(file)
        logging.info(f"Settings loaded from {settings_path}")
        return settings
    except Exception as e:
        logging.error(f"Error loading settings from {settings_path}: {str(e)}")
        raise

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        else:
            logging.info(f"Directory already exists: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {str(e)}")
        raise

def clean_review(review_text):
    try:
        review_text = re.sub(r'http\S+', '', review_text)
        review_text = re.sub('[^a-zA-z]', ' ', review_text)
        review_text = review_text.lower()
        tokens = word_tokenize(review_text)
        stop_words_set = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        stem_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_set and word not in ['film', 'movie', 'br', 'one']]
        return ' '.join(stem_tokens)
    except Exception as e:
        logging.error(f"Error cleaning review text: {str(e)}")
        raise

def prepare_data(file_path, settings, vectorizer=None, is_train_data=True):
    try:
        logging.info(f"Preparing data from {file_path}...")
        data = pd.read_csv(file_path)
        data['CleanReview_Stem'] = data['review'].apply(clean_review)

        if is_train_data:
            vectorizer = TfidfVectorizer(max_features=5000)
            X_tfidf_stem = vectorizer.fit_transform(data['CleanReview_Stem'])
            vectorizer_path = os.path.join(settings['general']['outputs_dir'], 'vectors')
            create_directory(vectorizer_path)
            with open(os.path.join(vectorizer_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)

            data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        else:
            with open(os.path.join(settings['general']['outputs_dir'], 'vectors', 'tfidf_vectorizer.pkl'), 'rb') as f:
                vectorizer = pickle.load(f)
            X_tfidf_stem = vectorizer.transform(data['CleanReview_Stem'])

        logging.info("TF-IDF vectorization complete.")

        processed_data_path = os.path.join(settings['general']['data_dir'], 'processed')
        create_directory(processed_data_path)

        tfidf_df = pd.DataFrame(X_tfidf_stem.toarray(), columns=vectorizer.get_feature_names_out())

        if is_train_data:
            tfidf_df['sentiment'] = data['sentiment']
            suffix = 'train'
        else:
            if 'sentiment' in data.columns:
                tfidf_df['sentiment'] = data['sentiment']
            suffix = 'inference'

        logging.info(f"Processed data saving...")
        output_file = os.path.join(processed_data_path, f'processed_{suffix}.csv')
        tfidf_df.to_csv(output_file, index=False)
        logging.info(f"Processed data saved to {output_file}")
    except Exception as e:
        logging.error(f"Error preparing data from {file_path}: {str(e)}")
        raise

try:
    settings = load_settings()

    if settings['data']['prepare_train_data']:
        train_file_path = os.path.join(settings['general']['data_dir'], 'raw', settings['data']['train_file_to_preparation'])
        prepare_data(train_file_path, settings, is_train_data=True)

    if settings['data']['prepare_inference_data']:
        inference_file_path = os.path.join(settings['general']['data_dir'], 'raw', settings['data']['inference_file_to_preparation'])
        prepare_data(inference_file_path, settings, is_train_data=False)

except Exception as e:
    logging.error(f"An unexpected error occurred: {str(e)}")
