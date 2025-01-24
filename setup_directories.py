#!/usr/bin/env python3
import os
import datetime
import json
from enum import Enum
import sys


class DirectoryStructure(Enum):
    CUTTINGS = "CUTTINGS"
    POSITIVES = "POSITIVES"
    NEGATIVES = "NEGATIVES"
    DATA_AUGMENTED = "DATA_AUGMENTED"
    DATA_TRAIN_AND_TEST = "DATA_TRAIN_AND_TEST"
    GENERATED_MODELS = "GENERATED_MODELS"


class Constants(Enum):
    CUTTINGS_FILE_PREFIX = "CUT"
    CUTTINGS_CUT_EXTENSION = ".png"
    
    DATA_EXTENSION = '.dat'
    
    DATA_AUGMENTED_FILE_POSITIVE = "DATA_POSITIVE"
    DATA_AUGMENTED_FILE_NEGATIVE = "DATA_NEGATIVE"
    DATA_AUGMENTED_FILE_TEST_POSITIVE = "DATA_TEST_POSITIVE"
    DATA_AUGMENTED_FILE_TEST_NEGATIVE = "DATA_TEST_NEGATIVE"

    DATA_TRAIN_AND_TEST_FILE_DATA_TRAIN = "DATA_TRAIN"
    DATA_TRAIN_AND_TEST_FILE_DATA_TEST = "DATA_TEST"
    DATA_TRAIN_AND_TEST_FILE_LABEL_TRAIN = "LABEL_TRAIN"
    DATA_TRAIN_AND_TEST_FILE_LABEL_TEST = "LABEL_TEST"

    GENERATED_MODELS_HISTORY = 'HISTORY.npy'
    GENERATED_MODELS_MODEL = 'MODEL.keras'
    GENERATED_MODELS_WEIGHTS = 'MODEL.weights.h5'

class FileNameManager:
    def __init__(self, base_dir, timestamp=None):
        self.base_dir = base_dir
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _build_path(self, subdir, filename, extension):
        return os.path.join(self.base_dir, subdir, f"{self.timestamp}{filename}.{extension}")

    def get_model_file(self):
        return self._build_path(DirectoryStructure.GENERATED_MODELS.value, "MODEL", "keras")

    def get_weights_file(self):
        return self._build_path(DirectoryStructure.GENERATED_MODELS.value, "WEIGHTS", "h5")


class DirectoryManager:
    def __init__(self, base_dir, directory_structure):
        self.base_dir = base_dir
        self.structure = directory_structure

    def create_directories(self):
        os.makedirs(self.base_dir, exist_ok=True)
        for folder in self.structure:
            subdir_path = os.path.join(self.base_dir, folder.value)
            if folder == DirectoryStructure.POSITIVES or folder == DirectoryStructure.NEGATIVES:
                continue
            os.makedirs(subdir_path, exist_ok=True)
            if folder == DirectoryStructure.CUTTINGS:
                os.makedirs(os.path.join(subdir_path, DirectoryStructure.POSITIVES.value), exist_ok=True)
                os.makedirs(os.path.join(subdir_path, DirectoryStructure.NEGATIVES.value), exist_ok=True)


class ConfigManager:
    def __init__(self, base_dir, timestamp):
        self.base_dir = base_dir
        self.timestamp = timestamp

    def save_config(self, file_data):
        config_path = os.path.join(self.base_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(file_data, f, indent=4)
        print(f"Generated files saved to {config_path}")


class DirectorySetupManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_data = {}

    def setup(self):
        # Create directories
        directory_manager = DirectoryManager(self.base_dir, DirectoryStructure)
        directory_manager.create_directories()
        data = {}
        data.update(self.build_dir_path())
        for item in Constants:
            data[item.name] = item.value
        self.file_data = data

        # Save configuration and file names
        config_manager = ConfigManager(self.base_dir, self.timestamp)
        config_manager.save_config(self.file_data)

    def get_file_data(self):
        return self.file_data
    
    def build_dir_path(self):

        return {
            'CUTTINGS_POSITIVES': os.path.abspath(os.path.join(
                self.base_dir,
                DirectoryStructure.CUTTINGS.value,
                DirectoryStructure.POSITIVES.value
            )),
            'CUTTINGS_NEGATIVES': os.path.abspath(os.path.join(
                self.base_dir,
                DirectoryStructure.CUTTINGS.value,
                DirectoryStructure.NEGATIVES.value
            )),
            'DATA_AUGMENTED': os.path.abspath(os.path.join(self.base_dir, DirectoryStructure.DATA_AUGMENTED.value)),
            'DATA_TRAIN_AND_TEST': os.path.abspath(os.path.join(self.base_dir, DirectoryStructure.DATA_TRAIN_AND_TEST.value)),
            'GENERATED_MODELS': os.path.abspath(os.path.join(self.base_dir, DirectoryStructure.GENERATED_MODELS.value)),
        }


if __name__ == "__main__":
    # Initialize and execute the setup
    new_dir = None
    if len(sys.argv) > 1:
        new_dir = sys.argv[1] or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join('__tests', new_dir)
    setup_manager = DirectorySetupManager(base_dir)
    setup_manager.setup()