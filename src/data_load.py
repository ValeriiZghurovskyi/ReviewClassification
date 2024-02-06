import os
import json
import requests
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings(settings_path='settings.json'):
    try:
        with open(settings_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading settings from {settings_path}: {str(e)}")
        raise

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {str(e)}")
        raise

def download_and_unzip(url, extract_to, expected_csv_name, save_as):
    local_zip = os.path.join(extract_to, 'temp.zip')
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_zip, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip)

        # Delete a file if it already exists
        csv_save_path = os.path.join(extract_to, save_as)
        if os.path.exists(csv_save_path):
            os.remove(csv_save_path)

        # Move CSV files and delete folders
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for name in files:
                if name == expected_csv_name:
                    shutil.move(os.path.join(root, name), csv_save_path)
            for name in dirs:
                shutil.rmtree(os.path.join(root, name))

        logging.info(f"Downloaded and unzipped data from {url}. Saved CSV as {csv_save_path}")
    except Exception as e:
        logging.error(f"Error downloading and unzipping data from {url}: {str(e)}")
        raise

try:
    settings = load_settings()

    data_dir = settings['general']['data_dir']
    raw_data_dir = os.path.join(data_dir, 'raw')
    create_directory(raw_data_dir)

    train_data_url = settings['data']['train_data_url']
    inference_data_url = settings['data']['inference_data_url']

    download_and_unzip(train_data_url, raw_data_dir, 'train.csv', 'train.csv')

    download_and_unzip(inference_data_url, raw_data_dir, 'test.csv', 'inference.csv')

except Exception as e:
    logging.error(f"An unexpected error occurred: {str(e)}")
