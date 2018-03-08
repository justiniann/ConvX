from keras import applications
from keras import backend as K
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from PIL import Image

from convx.convx_utils import *

RES_PATH = "..{0}..{0}resources{0}".format(os.path.sep)
IMG_PATH = "..{0}..{0}images{0}".format(os.path.sep)
BOTTLENECK_PATH = "..{0}..{0}bottleneck{0}".format(os.path.sep)
SAVE_PATH = "..{0}..{0}saved_models{0}".format(os.path.sep)

TRAIN_PATH = os.path.join(IMG_PATH, "train")
VAL_PATH = os.path.join(IMG_PATH, "validation")
TEST_PATH = os.path.join(IMG_PATH, "test")


def f3_score(y_true, y_pred, threshold_shift=0):
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


def build_fully_connected_top_layer(connecting_shape):
    top_layers = Sequential()
    top_layers.add(Flatten(input_shape=connecting_shape))
    top_layers.add(Dense(256, activation='relu'))
    top_layers.add(Dropout(0.05))
    top_layers.add(Dense(2, activation='softmax'))
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
    train_labels = []
    for i in range(0, healthy_train_images):
        train_labels.append(np.asarray([1, 0]))
    for i in range(0, unhealthy_train_images):
        train_labels.append(np.asarray([0, 1]))
    train_labels = np.asarray(train_labels)

    # load validation data
    validation_data = np.load(open(os.path.join(path_to_bottleneck_features, "validation.npy"), 'rb'))
    validation_labels = []
    for i in range(0, healthy_validation_images):
        validation_labels.append(np.asarray([1, 0]))
    for i in range(0, unhealthy_validation_images):
        validation_labels.append(np.asarray([0, 1]))
    validation_labels = np.asarray(validation_labels)

    top_layer = build_fully_connected_top_layer(train_data.shape[1:])

    top_layer.compile(loss='hinge',
                      optimizer="adam",
                      metrics=['accuracy'])

    top_layer.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                  verbose=1)
    return top_layer


def do_fine_tune_learning(base_model, top_layer, num_training_steps, num_validation_steps,
                          layers_to_train, epochs=20, batch_size=32, target_image_size=(250, 250)):
    model_for_finetune = Model(inputs=base_model.input, outputs=top_layer(base_model.output))

    for layer in model_for_finetune.layers[:len(model_for_finetune.layers) - layers_to_train]:
        layer.trainable = False

    model_for_finetune.compile(loss='hinge',
                               optimizer="adam",
                               metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = train_datagen.flow_from_directory(
        VAL_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary')

    model_for_finetune.fit_generator(
        train_generator,
        steps_per_epoch=num_training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_steps,
        verbose=1
    )

    return model_for_finetune


def build_dat_model(model_name, base_model, batch_size=32, target_image_size=(250, 250), transfer_epochs=10,
                    fine_tune_epochs=1, save=True):
    path_to_bottleneck_features = os.path.join(BOTTLENECK_PATH, model_name)
    path_to_saved_data = os.path.join(SAVE_PATH, model_name)

    healthy_train_images = count_files(os.path.join(TRAIN_PATH, "healthy"))
    unhealthy_train_images = count_files(os.path.join(TRAIN_PATH, "unhealthy"))
    healthy_validation_images = count_files(os.path.join(VAL_PATH, "healthy"))
    unhealthy_validation_images = count_files(os.path.join(VAL_PATH, "unhealthy"))

    num_training_steps = get_iterations_per_epoch((healthy_train_images + unhealthy_train_images), batch_size)
    num_validation_steps = get_iterations_per_epoch((healthy_validation_images + unhealthy_validation_images),
                                                    batch_size)

    # ***BOTTLENECK FEATURE EXTRACTION*********************************************************************************
    if os.path.exists(path_to_bottleneck_features) is False:
        print("Creating bottleneck features. This could take a while...")
        save_bottleneck_features(base_model,
                                 TRAIN_PATH,
                                 VAL_PATH,
                                 (num_training_steps, num_validation_steps),
                                 path_to_bottleneck_features,
                                 target_image_size=target_image_size)
        print("Bottleneck features saved!")

    # ***TRANSFER LEARNING*********************************************************************************************
    print("Transfer learning...")
    build_dir_path(path_to_saved_data)
    top_model_weights_path = os.path.join(path_to_saved_data, "transfer_learning_weights.h5")
    # Using the bottleneck features, train a fully connected classification layer (this will be the top layer)
    top_layer_for_transfer_learning = do_transfer_learning(path_to_bottleneck_features,
                                                           healthy_train_images,
                                                           unhealthy_train_images,
                                                           healthy_validation_images,
                                                           unhealthy_validation_images,
                                                           epochs=transfer_epochs,
                                                           batch_size=batch_size)

    # save the weights from the 'bottleneck model'
    top_layer_for_transfer_learning.save_weights(top_model_weights_path)
    print("Transfer learning complete!")

    # ***FINETUNE LEARNING*********************************************************************************************
    print("Beginning Fine-Tune learning...")
    top_layer = build_fully_connected_top_layer(base_model.output_shape[1:])
    top_layer.load_weights(top_model_weights_path)
    final_model = do_fine_tune_learning(
        base_model,
        top_layer,
        num_training_steps,
        num_validation_steps,
        layers_to_train=10,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        target_image_size=target_image_size)
    print("Fine tune learning complete!")
    if save:
        save_model(model_name, final_model, path_to_saved_data)
    return final_model


def evaluate_model(model, batch_size=32, target_image_size=(250, 250)):
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(momentum=0.95),
                  metrics=['accuracy', f3_score])

    healthy_test_images = count_files(os.path.join(TEST_PATH, "healthy"))
    unhealthy_test_images = count_files(os.path.join(TEST_PATH, "unhealthy"))

    test_iteration_count = int(get_iterations_per_epoch((healthy_test_images + unhealthy_test_images), batch_size))

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )

    test_y_actual = []
    test_y_predicted = []
    for i in range(0, test_iteration_count):
        data = test_generator.next()
        test_y_actual.append(data[1])
        preds = model.predict(data[0], batch_size=batch_size)
        test_y_predicted.append(preds)
        print(preds)

    test_y_actual = np.asarray(test_y_actual)


if __name__ == '__main__':
    model = build_dat_model("convx_model", applications.ResNet50(include_top=False, input_shape=(250,250,3)), batch_size=32, fine_tune_epochs=20, transfer_epochs=1)
