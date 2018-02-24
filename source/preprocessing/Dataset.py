#Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],

import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool


RES_PATH = "..{0}..{0}resources{0}".format(os.path.sep)
IMG_PATH = "..{0}..{0}images{0}".format(os.path.sep)
train_path = "{}{}{}".format(IMG_PATH, "train", os.path.sep)
validation_path = "{}{}{}".format(IMG_PATH, "validation", os.path.sep)
test_path = "{}{}{}".format(IMG_PATH, "test", os.path.sep)


def move_images():
    train_X, train_y, validation_X, validation_y, test_X, test_y = split_that_shit()
    # train
    for i in range(0, len(train_X)):
        file_name = train_X[i]
        classification_directory = "healthy" if np.argmax(train_y[i]) == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(train_path, classification_directory, os.path.sep, file_name))

    # validation
    for i in range(0, len(validation_X)):
        file_name = validation_X[i]
        classification_directory = "healthy" if np.argmax(validation_y[i]) == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(validation_path, classification_directory, os.path.sep, file_name))

    # test
    for i in range(0, len(test_X)):
        file_name = test_X[i]
        classification_directory = "healthy" if np.argmax(test_y[i]) == 0 else "unhealthy"
        os.rename("{}{}".format(IMG_PATH, file_name), "{}{}{}{}"
                  .format(test_path, classification_directory, os.path.sep, file_name))


def chunk_list(images, chuncks):
    random.shuffle(images)
    size = len(images) / chuncks
    return [images[i:i+size] for i in range(0, len(images), size)]


def read_data_entry_csv():
    return pd.read_csv("{}Data_Entry_2017.csv".format(RES_PATH))


def read_data_entry_preprocessed_csv(size=None):
    df = pd.read_csv("{}Preprocessed_Data_Entry_2017.csv".format(RES_PATH))
    if size:
        df = df[:size]
    return df


def reclassify_findings_column(row):
    return 0 if row['Finding Labels'] == "No Finding" else 1


def vectorize_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_vector = image.img_to_array(img)
    return np.expand_dims(img_vector, axis=0)


def vectorize_images(file_names):
    if len(file_names) == 0:
        return np.ndarray([])
    list_of_vectors = [vectorize_img("{}{}".format(IMG_PATH, i)) for i in file_names]
    return np.vstack(list_of_vectors)


def preprocess():
    dataset = read_data_entry_csv()
    dataset["Health Flag"] = dataset.apply(reclassify_findings_column, axis=1)
    preprocessed_dataset = dataset[['Image Index', 'Health Flag']].copy()
    preprocessed_dataset.to_csv("{}Preprocessed_Data_Entry_2017.csv".format(RES_PATH), sep=',', index=False)


def split_that_shit(size=None, test_percentage=0.2, validation_percentage=0.1):
    dataset = read_data_entry_preprocessed_csv(size=size)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_percentage)
    return X_train.flatten(), format_dependent(y_train), X_validation.flatten(), format_dependent(y_validation), X_test.flatten(), format_dependent(y_test)


def format_dependent(y, categories=2):
    return np_utils.to_categorical(y, categories)


def ouptut_distribution():
    dataset = read_data_entry_csv()
    value_counts = dataset['Finding Labels'].value_counts()
    num_images = len(dataset['Finding Labels'])
    num_healthy_images = value_counts['No Finding']
    print("Total number of images: {}".format(num_images))
    print("Total number of healthy images: {}".format(num_healthy_images))
    print("Total number of unhealthy images: {}".format(num_images - num_healthy_images))


if __name__ == '__main__':
    move_images()
