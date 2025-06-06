import sys
import os
import kagglehub
import shutil
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import *
import zipfile

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, dataset_name: str, target_dir: str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        raw_dir = os.path.join(self.target_dir, 'raw')
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created the {raw_dir}")
            except Exception as e:
                logger.error(f"Error creating directory {raw_dir}: {e}")
                raise CustomException(f"Error creating directory {raw_dir}: {e}")
        return raw_dir
    
    def extract_images_and_labels(self, path: str, raw_dir: str):
        try:
            if path.endswith('.zip'):
                logger.info("Extracting zip file")
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)
            
            images_folder = os.path.join(path, 'Images')
            labels_folder = os.path.join(path, 'Labels')
            logger.info(f"Image and labels folder : {images_folder}, {labels_folder}")

            if os.path.exists(images_folder):
                shutil.move(images_folder, os.path.join(raw_dir, 'Images'))
                logger.info("Images move succesffully....")
            else:
                logger.info("Images folder dont exists...")

            if os.path.exists(labels_folder):
                shutil.move(labels_folder, os.path.join(raw_dir, 'Labels'))
                logger.info("Labels move succesffully....")
            else:
                logger.info("Labels folder dont exists...")

        except Exception as e:
            logger.error(f"Error extracting images and labels: {e}")
            raise CustomException(f"Error extracting images and labels: {e}")
    
    def download_dataset(self, raw_dir: str):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to {path}")

            self.extract_images_and_labels(path, raw_dir)
        
        except Exception as e:
            logger.error(f"Error downloading dataset {self.dataset_name}: {e}")
            raise CustomException(f"Error downloading dataset {self.dataset_name}: {e}")
        
    def run(self):
        try:
            logger.info("Starting data ingestion process...")

            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)

            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    data_ingestion = DataIngestion(dataset_name=DATASET_NAME, target_dir=TARGET_DIR)
    data_ingestion.run()
# This code is for data ingestion, downloading a dataset from Kaggle, and extracting images and labels.     