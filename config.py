#!/usr/bin/env python3

import json

class Config:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as file:
            json_dict = json.load(file)

        self.CUTTINGS_POSITIVES = json_dict.get('CUTTINGS_POSITIVES', '')
        self.CUTTINGS_NEGATIVES = json_dict.get('CUTTINGS_NEGATIVES', '')
        self.DATA_AUGMENTED = json_dict.get('DATA_AUGMENTED', '')
        self.DATA_TRAIN_AND_TEST = json_dict.get('DATA_TRAIN_AND_TEST', '')
        self.GENERATED_MODELS = json_dict.get('GENERATED_MODELS', '')
        self.CUTTINGS_FILE_PREFIX = json_dict.get('CUTTINGS_FILE_PREFIX', '')
        self.CUTTINGS_CUT_EXTENSION = json_dict.get('CUTTINGS_CUT_EXTENSION', '')
        self.DATA_AUGMENTED_FILE_POSITIVE = json_dict.get('DATA_AUGMENTED_FILE_POSITIVE', '')
        self.DATA_AUGMENTED_FILE_NEGATIVE = json_dict.get('DATA_AUGMENTED_FILE_NEGATIVE', '')
        self.DATA_AUGMENTED_FILE_TEST_POSITIVE = json_dict.get('DATA_AUGMENTED_FILE_TEST_POSITIVE', '')
        self.DATA_AUGMENTED_FILE_TEST_NEGATIVE = json_dict.get('DATA_AUGMENTED_FILE_TEST_NEGATIVE', '')
        self.DATA_EXTENSION = json_dict.get('DATA_EXTENSION', '')
        self.DATA_TRAIN_AND_TEST_FILE_DATA_TRAIN = json_dict.get('DATA_TRAIN_AND_TEST_FILE_DATA_TRAIN', '')
        self.DATA_TRAIN_AND_TEST_FILE_DATA_TEST = json_dict.get('DATA_TRAIN_AND_TEST_FILE_DATA_TEST', '')
        self.DATA_TRAIN_AND_TEST_FILE_LABEL_TRAIN = json_dict.get('DATA_TRAIN_AND_TEST_FILE_LABEL_TRAIN', '')
        self.DATA_TRAIN_AND_TEST_FILE_LABEL_TEST = json_dict.get('DATA_TRAIN_AND_TEST_FILE_LABEL_TEST', '')
        self.GENERATED_MODELS_HISTORY = json_dict.get('GENERATED_MODELS_HISTORY', '')
        self.GENERATED_MODELS_MODEL = json_dict.get('GENERATED_MODELS_MODEL', '')
        self.GENERATED_MODELS_WEIGHTS = json_dict.get('GENERATED_MODELS_WEIGHTS', '')


config = None

def load_config(json_file_path):
    global config
    print('=== 1')
    if config is None:
        print('=== 2')
        config = Config(json_file_path)
        print('=== 3', config.CUTTINGS_POSITIVES)
    return config
