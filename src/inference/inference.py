import os
import pickle
import json
import pandas as pd
import logging

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

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def create_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {str(e)}")
        raise

def infer(model, data, outputs_dir):
    try:
        if 'sentiment' in data.columns:
            data = data.drop('sentiment', axis=1)

        y_pred = model.predict(data)

        predictions_dir = os.path.join(outputs_dir, 'predictions')
        create_directory(predictions_dir)

        results_path = os.path.join(predictions_dir, 'inference_results.csv')
        results_df = pd.DataFrame({'prediction': y_pred})
        results_df.to_csv(results_path, index=False)
        logging.info(f"Inference results saved to {results_path}")
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        raise

def main():
    try:
        settings = load_settings()

        logging.info("Loading data for inference...")
        inference_data_path = os.path.join(settings['general']['data_dir'], 'processed', settings['inference']['inp_table_name'])
        inference_data = pd.read_csv(inference_data_path)

        logging.info("Downloading the model...")
        model_path = os.path.join(settings['general']['outputs_dir'], settings['general']['models_dir'], settings['inference']['model_name'])
        model = load_model(model_path)

        logging.info("Starting an inference...")
        infer(model, inference_data, settings['general']['outputs_dir'])
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()