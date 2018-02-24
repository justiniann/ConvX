import time
from keras import applications
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator

from preprocessing.Dataset import *

RES_PATH = "..{0}..{0}resources{0}".format(os.path.sep)
IMG_PATH = "..{0}..{0}images{0}".format(os.path.sep)
BOTTLENECK_PATH = "..{0}..{0}bottleneck{0}".format(os.path.sep)
SAVE_PATH = "..{0}..{0}saved_models{0}".format(os.path.sep)

TRAIN_PATH = "{}{}{}".format(IMG_PATH, "train", os.path.sep)
VAL_PATH = "{}{}{}".format(IMG_PATH, "validation", os.path.sep)


def build_dir_path(path):
    if not os.path.exists(path):  # ensure the directory is there, create it if you have to
        os.makedirs(path)


def count_files(root_dir):
    return sum([len(files) for r, d, files in os.walk(root_dir)])


def get_iterations_per_epoch(total_images, batch_size):
    return np.ceil(total_images / batch_size)


def f5_score(y_true, y_pred, threshold_shift=0):
    """
    Calculate fbeta score for Keras metrics.
    from https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras
    """
    beta = 3

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    fbeta = ((beta_squared + 1) * (precision * recall) /
             (beta_squared * precision + recall + K.epsilon()))
    return fbeta


def save_model(model_name, model, save_path):
    # serialize model to JSON
    model_json = model.to_json()
    build_dir_path(save_path)
    with open(os.path.join(save_path, "{}.json".format(model_name)), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(save_path, "{}_weights.h5".format(model_name)))
    print("Saved model to disk")


def build_fully_connected_top_layer(connecting_shape):
    top_layers = Sequential()
    top_layers.add(Flatten(input_shape=connecting_shape))
    top_layers.add(Dense(256, activation='relu'))
    top_layers.add(Dropout(0.2))
    top_layers.add(Dense(1, activation='sigmoid'))
    return top_layers


def save_bottleneck_features(model, train_path, validation_path, step_counts, bottleneck_file_path,
                             batch_size=32, target_image_size=(250, 250)):
    train_bottleneck_file = os.path.join(bottleneck_file_path, "train.npy")
    validation_bottleneck_file = os.path.join(bottleneck_file_path, "validation.npy")

    data_generator = ImageDataGenerator(rescale=1. / 255)

    train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    bottleneck_features_train = model.predict_generator(train_generator, step_counts[0])
    build_dir_path(bottleneck_file_path)
    np.save(open(train_bottleneck_file, 'wb'), bottleneck_features_train)

    validation_path_generator = data_generator.flow_from_directory(
        validation_path,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    bottleneck_features_validation = model.predict_generator(validation_path_generator, step_counts[1])
    np.save(open(validation_bottleneck_file, 'wb'), bottleneck_features_validation)


def do_transfer_learning(path_to_bottleneck_features, healthy_train_images, unhealthy_train_images,
                         healthy_validation_images, unhealthy_validation_images, epochs=20, batch_size=32):
    # load training data
    train_data = np.load(open(os.path.join(path_to_bottleneck_features, "train.npy"), 'rb'))
    train_labels = np.array([0] * healthy_train_images + [1] * unhealthy_train_images)

    # load validation data
    validation_data = np.load(open(os.path.join(path_to_bottleneck_features, "validation.npy"), 'rb'))
    validation_labels = np.array([0] * healthy_validation_images + [1] * unhealthy_validation_images)

    top_layer = build_fully_connected_top_layer(train_data.shape[1:])

    top_layer.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    top_layer.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))
    return top_layer


def do_fine_tune_learning(base_model, top_layer, model_weights_save_file, num_training_steps,num_validation_steps,
                          layers_to_train, epochs=20, batch_size=32, target_image_size=(250, 250)):
    model_for_finetune = Model(inputs=base_model.input, outputs=top_layer(base_model.output))

    for layer in model_for_finetune.layers[:len(model_for_finetune.layers) - layers_to_train]:
        layer.trainable = False

    model_for_finetune.compile(loss='binary_crossentropy',
                               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                               metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
        model_weights_save_file,
        monitor='val_loss',
        save_best_only=True
    )

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model_for_finetune.fit_generator(
        train_generator,
        steps_per_epoch=num_training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_steps,
        callbacks=[checkpointer],
        verbose=1
    )
    return model_for_finetune


def build_dat_model(model_name, base_model, epochs=20, batch_size=32, target_image_size=(250, 250),
                    get_bottleneck_features=False, transfer_learn=True, save=True):
    path_to_bottleneck_features = os.path.join(BOTTLENECK_PATH, model_name)
    path_to_saved_data = os.path.join(SAVE_PATH, model_name)

    healthy_train_images = count_files(os.path.join(train_path, "healthy"))
    unhealthy_train_images = count_files(os.path.join(train_path, "unhealthy"))
    healthy_validation_images = count_files(os.path.join(validation_path, "healthy"))
    unhealthy_validation_images = count_files(os.path.join(validation_path, "unhealthy"))

    num_training_steps = get_iterations_per_epoch((healthy_train_images + unhealthy_train_images), batch_size)
    num_validation_steps = get_iterations_per_epoch((healthy_validation_images + unhealthy_validation_images),
                                                    batch_size)

    # ***BOTTLENECK FEATURE EXTRACTION*********************************************************************************
    if get_bottleneck_features or os.path.exists(path_to_bottleneck_features) is False:
        print("Creating bottleneck features. This could take a while...")
        save_bottleneck_features(base_model,
                                 TRAIN_PATH,
                                 VAL_PATH,
                                 (num_training_steps, num_validation_steps),
                                 path_to_bottleneck_features,
                                 target_image_size=target_image_size)
        print("Bottleneck features saved!")
    else:
        print("Using bottleneck features from previously trained model")

    # ***TRANSFER LEARNING*********************************************************************************************
    print("Transfer learning...")
    top_model_weights_path = os.path.join(path_to_saved_data, "transfer_learning_weights.h5")
    if transfer_learn or os.path.exists(top_model_weights_path) is False:
        # Using the bottleneck features, train a fully connected classification layer (this will be the top layer)
        top_layer_for_transfer_learning = do_transfer_learning(path_to_bottleneck_features,
                                                               healthy_train_images,
                                                               unhealthy_train_images,
                                                               healthy_validation_images,
                                                               unhealthy_validation_images,
                                                               epochs=epochs,
                                                               batch_size=batch_size)

        # save the weights from the 'bottleneck model'
        build_dir_path(path_to_saved_data)
        top_layer_for_transfer_learning.save_weights(top_model_weights_path)
        print("Transfer learning complete!")
    else:
        print("Using top layer weights from previously trained model")

    # ***FINETUNE LEARNING*********************************************************************************************
    print("Beginning Fine-Tune learning...")
    top_layer = build_fully_connected_top_layer(base_model.output_shape[1:])
    top_layer.load_weights(top_model_weights_path)
    final_model = do_fine_tune_learning(
        base_model,
        top_layer,
        path_to_saved_data,
        num_training_steps,
        num_validation_steps,
        layers_to_train=2,
        epochs=epochs,
        batch_size=batch_size,
        target_image_size=target_image_size)
    print("Fine tune learning complete!")
    if save:
        save_model(model_name, final_model, path_to_saved_data)
    return final_model


if __name__ == '__main__':
    start_time = time.time()
    model = build_dat_model("VGG16",
                            applications.VGG16(include_top=False, weights='imagenet', input_shape=(250, 250, 3)),
                            epochs=1,
                            save=True)
    print("Training complete, total runtime was {} sec".format(time.time()-start_time))
