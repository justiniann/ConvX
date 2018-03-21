import os

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

RES_PATH = "..{0}..{0}resources{0}".format(os.path.sep)
BOTTLENECK_PATH = "..{0}..{0}bottleneck{0}".format(os.path.sep)
SAVE_PATH = "..{0}..{0}saved_models{0}".format(os.path.sep)

IMG_PATH = "..{0}..{0}images{0}".format(os.path.sep)
TRAIN_PATH = os.path.join(IMG_PATH, "train")
VAL_PATH = os.path.join(IMG_PATH, "validation")
TEST_PATH = os.path.join(IMG_PATH, "test")


def read_data_entry_csv():
    return pd.read_csv("{}Data_Entry_2017.csv".format(RES_PATH))


def read_data_entry_preprocessed_csv():
    return pd.read_csv("{}Preprocessed_Data_Entry_2017.csv".format(RES_PATH))


def build_dir_path(path):
    if not os.path.exists(path):  # ensure the directory is there, create it if you have to
        os.makedirs(path)


def count_files(root_dir):
    return sum([len(files) for r, d, files in os.walk(root_dir)])


def get_iterations_per_epoch(total_images, batch_size):
    return np.ceil(total_images / batch_size)


def format_dependent(y, categories=2):
    return np_utils.to_categorical(y, categories)

def get_img_filepath(filename, y):
    classification = "healthy" if y == 0 else "unhealthy"
    if os.path.exists(os.path.join(TRAIN_PATH, classification, filename)):
        return os.path.join(TRAIN_PATH, classification, filename)
    elif os.path.exists(os.path.join(TEST_PATH, classification, filename)):
        return os.path.join(TEST_PATH, classification, filename)
    elif os.path.exists(os.path.join(VAL_PATH, classification, filename)):
        return os.path.join(VAL_PATH, classification, filename)

def split_image_groups(test_percentage=0.2, validation_percentage=0.1):
    dataset = read_data_entry_preprocessed_csv()
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_percentage)
    return X_train.flatten(), y_train, X_validation.flatten(), y_validation, X_test.flatten(), y_test


def preprocess():
    preprocessed_csv_file = "{}Preprocessed_Data_Entry_2017.csv".format(RES_PATH)
    if not os.path.exists(preprocessed_csv_file):
        dataset = read_data_entry_csv()
        dataset["Health Flag"] = dataset.apply(lambda row: 0 if row['Finding Labels'] == "No Finding" else 1, axis=1)
        preprocessed_dataset = dataset[['Image Index', 'Health Flag']].copy()
        preprocessed_dataset.to_csv(preprocessed_csv_file, sep=',', index=False)
    print("Data successfully preprocessed!")


def move_images():
    train_X, train_y, validation_X, validation_y, test_X, test_y = split_image_groups()
    # train
    for i in range(0, len(train_X)):
        file_name = train_X[i]
        classification_directory = "healthy" if train_y[i] == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(TRAIN_PATH, classification_directory, os.path.sep, file_name))

    # validation
    for i in range(0, len(validation_X)):
        file_name = validation_X[i]
        classification_directory = "healthy" if validation_y[i] == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(VAL_PATH, classification_directory, os.path.sep, file_name))

    # test
    for i in range(0, len(test_X)):
        file_name = test_X[i]
        classification_directory = "healthy" if test_y[i] == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(TEST_PATH, classification_directory, os.path.sep, file_name))


def ouptut_distribution():
    dataset = read_data_entry_csv()
    value_counts = dataset['Finding Labels'].value_counts()
    num_images = len(dataset['Finding Labels'])
    num_healthy_images = value_counts['No Finding']
    print("Total number of images: {}".format(num_images))
    print("Total number of healthy images: {}".format(num_healthy_images))
    print("Total number of unhealthy images: {}".format(num_images - num_healthy_images))


def get_image_intensity():
    dataset = read_data_entry_csv()
    counts = [0] * 10
    for label in dataset['Finding Labels']:
        if label == "No Finding":
            counts[0] += 1
        else:
            num_diseases = len(label.split('|'))
            counts[num_diseases] += 1
    print("Disease counts: {}".format(counts))

if __name__ == '__main__':
    preprocess()
    move_images()
    get_image_intensity()
