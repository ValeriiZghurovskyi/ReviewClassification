import os
import json
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings(settings_path='settings.json'):
    try:
        with open(settings_path, 'r') as file:
            settings = json.load(file)
        logging.info(f"Settings loaded from {settings_path}")
        return settings
    except Exception as e:
        logging.error(f"Error loading settings from {settings_path}: {str(e)}")
        raise

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def train_model(X_train, y_train, C_value):
    try:
        model = LinearSVC(C=C_value, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error training the model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Recall: {recall}")
        logging.info(f"Precision: {precision}")
        logging.info(f"F1 Score: {f1}")
        logging.info("\nClassification report:\n" + classification_rep)

        return accuracy, recall, precision, f1, classification_rep
    except Exception as e:
        logging.error(f"Error evaluating the model: {str(e)}")
        raise

def save_model(model, model_path):
    try:
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {str(e)}")
        raise

def save_metrics(accuracy, recall, precision, f1, classification_rep, metrics_path):
    try:
        with open(metrics_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write("\nClassification report:\n" + classification_rep)
        logging.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
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

def main():
    try:
        settings = load_settings()

        logging.info("Reading data...")
        processed_data_path = os.path.join(settings['general']['data_dir'], 'processed')
        train_data_path = os.path.join(processed_data_path, f"{settings['train']['table_name'].replace('.csv', '')}.csv")
        data = load_data(train_data_path)

        X_train, X_test, y_train, y_test = train_test_split(data.drop('sentiment', axis=1), data['sentiment'], test_size=settings['train']['test_size'], random_state=settings['general']['random_state'])

        logging.info("Training the model...")
        best_model = train_model(X_train, y_train, settings['train']['C_value'])

        logging.info("Evaluating the model...")
        accuracy, recall, precision, f1, classification_rep = evaluate_model(best_model, X_test, y_test)

        model_name = settings['train']['model_name'].replace('.pkl', '') + '.pkl'
        model_path = os.path.join(settings['general']['outputs_dir'], 'models', model_name)
        save_model(best_model, model_path)

        metrics_dir = os.path.join(settings['general']['outputs_dir'], 'predictions')
        create_directory(metrics_dir)
        metrics_name = f"metrics_{model_name.replace('.pkl', '')}.txt"
        metrics_path = os.path.join(metrics_dir, metrics_name)
        save_metrics(accuracy, recall, precision, f1, classification_rep, metrics_path)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()