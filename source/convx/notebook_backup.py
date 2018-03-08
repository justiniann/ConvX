import os
from keras.applications import ResNet50
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import backend as K
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score
from convx_utils import *

if __name__ == '__main__':
    # This is our base model. We will use the weights and structure already known to be successful in other domains and
    #   adjust it to fit our current problem
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    target_image_size = (512, 512)
    batch_size = 32
    transfer_learning_epochs = 1
    fine_tuning_epochs = 1
    fine_tuning_layers_to_train = 10

    # The following are directories used for reading/saving data
    RES_PATH = "..{0}..{0}resources{0}".format(os.path.sep)
    IMG_PATH = "..{0}..{0}images{0}".format(os.path.sep)
    BOTTLENECK_PATH = "..{0}..{0}bottleneck{0}".format(os.path.sep)
    SAVE_PATH = "..{0}..{0}saved_models{0}".format(os.path.sep)
    TRAIN_PATH = os.path.join(IMG_PATH, "train")
    VAL_PATH = os.path.join(IMG_PATH, "validation")
    TEST_PATH = os.path.join(IMG_PATH, "test")

    model_name = "convx_model"  # The directory all data will be saved in will be named whatever this value is.
    models_save_directory = os.path.join(SAVE_PATH, model_name)
    build_dir_path(models_save_directory)  # build the directory structure we need for saving results

    def count_files(root_dir):
        return sum([len(files) for r, d, files in os.walk(root_dir)])

    def get_iterations_per_epoch(total_images, batch_size):
        return np.ceil(total_images / batch_size)

    healthy_train_images = count_files(os.path.join(TRAIN_PATH, "healthy"))
    unhealthy_train_images = count_files(os.path.join(TRAIN_PATH, "unhealthy"))
    healthy_validation_images = count_files(os.path.join(VAL_PATH, "healthy"))
    unhealthy_validation_images = count_files(os.path.join(VAL_PATH, "unhealthy"))

    num_training_steps = get_iterations_per_epoch((healthy_train_images + unhealthy_train_images), batch_size)
    num_validation_steps = get_iterations_per_epoch((healthy_validation_images + unhealthy_validation_images), batch_size)

    data_generator = ImageDataGenerator(rescale=1. / 255)

    bottleneck_file_path = os.path.join(BOTTLENECK_PATH, model_name)
    train_bottleneck_file = os.path.join(bottleneck_file_path, "train.npy")
    validation_bottleneck_file = os.path.join(bottleneck_file_path, "validation.npy")

    # # Extract bottleneck features if they have not been already
    # if not os.path.exists(bottleneck_file_path):
    #     data_generator = ImageDataGenerator(rescale=1. / 255)
    #
    # train_generator = data_generator.flow_from_directory(
    #     TRAIN_PATH,
    #     target_size=target_image_size,
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=False
    # )
    #
    # build_dir_path(bottleneck_file_path)
    # bottleneck_features_train = base_model.predict_generator(train_generator, num_training_steps)
    # np.save(open(train_bottleneck_file, 'wb'), bottleneck_features_train)
    #
    # validation_path_generator = data_generator.flow_from_directory(
    #     VAL_PATH,
    #     target_size=target_image_size,
    #     batch_size=batch_size,
    #     class_mode='binary',
    #     shuffle=False
    # )
    #
    # bottleneck_features_validation = base_model.predict_generator(validation_path_generator, num_validation_steps)
    # np.save(open(validation_bottleneck_file, 'wb'), bottleneck_features_validation)
    #
    #
    # def build_fully_connected_top_layer(connecting_shape):
    #     top_layers = Sequential()
    #     top_layers.add(Flatten(input_shape=connecting_shape))
    #     top_layers.add(Dense(256, activation='relu'))
    #     top_layers.add(Dropout(0.1))
    #     top_layers.add(Dense(1, activation='sigmoid'))
    #     return top_layers
    #
    #
    # # load training data
    # train_data = np.load(open(train_bottleneck_file, 'rb'))
    # train_labels = np.array(([0] * healthy_train_images) + ([1] * unhealthy_train_images))
    #
    # # load validation data
    # validation_data = np.load(open(validation_bottleneck_file, 'rb'))
    # validation_labels = np.array([0] * healthy_validation_images + [1] * unhealthy_validation_images)
    #
    # top_layer = build_fully_connected_top_layer(train_data.shape[1:])
    #
    # top_layer.compile(loss='hinge',
    #                   optimizer='adam',
    #                   metrics=['accuracy'])
    #
    # top_layer.fit(train_data, train_labels,
    #               epochs=transfer_learning_epochs,
    #               batch_size=batch_size,
    #               validation_data=(validation_data, validation_labels),
    #               verbose=1)
    #
    # top_layers_weights_path = os.path.join(models_save_directory, "transfer_learning_weights.h5")
    # top_layer.save_weights(top_layers_weights_path)
    #
    # top_layer = build_fully_connected_top_layer(base_model.output_shape[1:])
    # top_layer.load_weights(top_layers_weights_path)
    #
    # convx_model = Model(inputs=base_model.input, outputs=top_layer(base_model.output))
    # convx_model.compile(loss='binary_crossentropy',
    #                     optimizer=optimizers.SGD(momentum=0.95),
    #                     metrics=['accuracy'])
    #
    #
    # data_generator = ImageDataGenerator(rescale=1. / 255)
    #
    # convx_model = Model(inputs=base_model.input, outputs=top_layer(base_model.output))
    #
    # for layer in convx_model.layers[:len(convx_model.layers) - fine_tuning_layers_to_train]:
    #     layer.trainable = False
    #
    # convx_model.compile(loss='hinge',
    #                     optimizer='adam',
    #                     metrics=['accuracy'])
    #
    # data_generator = ImageDataGenerator(rescale=1. / 255)
    #
    # train_generator = data_generator.flow_from_directory(
    #     TRAIN_PATH,
    #     target_size=target_image_size,
    #     batch_size=batch_size,
    #     class_mode='binary')
    #
    # validation_generator = data_generator.flow_from_directory(
    #     VAL_PATH,
    #     target_size=target_image_size,
    #     batch_size=batch_size,
    #     class_mode='binary')
    #
    # convx_model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=num_training_steps,
    #     epochs=fine_tuning_epochs,
    #     validation_data=validation_generator,
    #     validation_steps=num_validation_steps,
    #     verbose=1
    # )

    # with open(os.path.join(models_save_directory, "best_model.json".format(model_name)), "w") as json_file:
    #     json_file.write(convx_model.to_json())


    def f3_score(y_true, y_pred, threshold_shift=0):
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

    convx_model = None
    with open(os.path.join(models_save_directory, "best_model.json"), 'r') as model_file:
        convx_model = model_from_json(model_file.read())

    convx_model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.SGD(momentum=0.95),
                        metrics=['accuracy', f3_score])

    healthy_test_images = count_files(os.path.join(TEST_PATH, "healthy"))
    unhealthy_test_images = count_files(os.path.join(TEST_PATH, "unhealthy"))
    test_iteration_count = get_iterations_per_epoch((healthy_test_images + unhealthy_test_images), batch_size)
    test_labels = np.array(([0] * healthy_test_images) + ([1] * unhealthy_test_images))

    data_generator = ImageDataGenerator(rescale=1. / 255)


    test_generator = data_generator.flow_from_directory(
        TEST_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )

    raw_predictions = convx_model.predict_generator(test_generator, test_iteration_count)
    predictions = np.round(raw_predictions)
    accuracy = accuracy_score(test_labels, predictions)
    f3_score = fbeta_score(test_labels, predictions, 3)
    conf_matrix = confusion_matrix(test_labels, predictions)
    print("Accuracy: {}".format(accuracy))
    print("F3 Score: {}".format(f3_score))
    print("Confusion Matrix: \n{}".format(conf_matrix))
