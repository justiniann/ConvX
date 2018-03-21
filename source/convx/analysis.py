from keras.applications import ResNet50
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix

from convx.convx_utils import *

if __name__ == '__main__':
    # This is our base model. We will use the weights and structure already known to be successful in other domains and
    #   adjust it to fit our current problem
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    target_image_size = (512, 512)
    batch_size = 32
    transfer_learning_epochs = 100
    fine_tuning_epochs = 20
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

    data_generator = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=20,
        height_shift_range=20
    )

    bottleneck_file_path = os.path.join(BOTTLENECK_PATH, model_name)
    train_bottleneck_file = os.path.join(bottleneck_file_path, "train.npy")
    validation_bottleneck_file = os.path.join(bottleneck_file_path, "validation.npy")

    # BOTTLENECK EXTRACTION -------------------------------------------------------------------------------------------
    if not os.path.exists(bottleneck_file_path):
        train_generator = data_generator.flow_from_directory(
            TRAIN_PATH,
            target_size=target_image_size,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False
        )

        bottleneck_features_train = base_model.predict_generator(train_generator, num_training_steps)
        build_dir_path(bottleneck_file_path)
        np.save(open(train_bottleneck_file, 'wb'), bottleneck_features_train)

        validation_path_generator = data_generator.flow_from_directory(
            VAL_PATH,
            target_size=target_image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        bottleneck_features_validation = base_model.predict_generator(validation_path_generator, num_validation_steps)
        np.save(open(validation_bottleneck_file, 'wb'), bottleneck_features_validation)

    # TRANSFER LEARNING -----------------------------------------------------------------------------------------------
    def build_fully_connected_top_layer(connecting_shape, dropout=True):
        top_layers = Sequential()
        # top_layers.add(Flatten(input_shape=connecting_shape))
        top_layers.add(GlobalAveragePooling2D(input_shape=connecting_shape))
        top_layers.add(Dense(256, activation='relu'))
        if dropout:
            top_layers.add(Dropout(0.1))
        top_layers.add(Dense(2, activation='softmax'))
        return top_layers

    # load training data
    train_data = np.load(open(train_bottleneck_file, 'rb'))
    train_labels = to_categorical(np.array(([0] * healthy_train_images) + ([1] * unhealthy_train_images)),
                                  num_classes=2)

    # load validation data
    validation_data = np.load(open(validation_bottleneck_file, 'rb'))
    validation_labels = to_categorical(np.array([0] * healthy_validation_images + [1] * unhealthy_validation_images),
                                       num_classes=2)

    top_layer = build_fully_connected_top_layer(train_data.shape[1:])

    def compile_model(model):
        model.compile(loss='hinge',
                      optimizer='sgd',
                      metrics=['accuracy'])

    compile_model(top_layer)

    top_layer.fit(train_data, train_labels,
                  epochs=transfer_learning_epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels),
                  verbose=1)

    top_layers_weights_path = os.path.join(models_save_directory, "transfer_learning_weights.h5")
    top_layer.save_weights(top_layers_weights_path)

    # FINE TUNING -----------------------------------------------------------------------------------------------------
    top_layer = build_fully_connected_top_layer(base_model.output_shape[1:])
    top_layer.load_weights(top_layers_weights_path)

    convx_model = Model(inputs=base_model.input, outputs=top_layer(base_model.output))

    for layer in convx_model.layers[:len(convx_model.layers) - fine_tuning_layers_to_train]:
        layer.trainable = False

    compile_model(convx_model)

    train_generator = data_generator.flow_from_directory(
        TRAIN_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        VAL_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='categorical')

    convx_model.fit_generator(
        train_generator,
        steps_per_epoch=num_training_steps,
        epochs=fine_tuning_epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_steps,
        verbose=1
    )

    with open(os.path.join(models_save_directory, "best_model.json".format(model_name)), "w") as json_file:
        json_file.write(convx_model.to_json())

    # MODEL EVALUATION ------------------------------------------------------------------------------------------------
    with open(os.path.join(models_save_directory, "best_model.json"), 'r') as model_file:
        convx_model = model_from_json(model_file.read())
    compile_model(convx_model)

    healthy_test_images = count_files(os.path.join(TEST_PATH, "healthy"))
    unhealthy_test_images = count_files(os.path.join(TEST_PATH, "unhealthy"))
    test_iteration_count = get_iterations_per_epoch((healthy_test_images + unhealthy_test_images), batch_size)

    # load test data
    test_labels = np.array(([0] * healthy_test_images) + ([1] * unhealthy_test_images))
    formatted_test_labels = to_categorical(test_labels, num_classes=2)

    test_generator = data_generator.flow_from_directory(
        TEST_PATH,
        target_size=target_image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )

    raw_predictions = convx_model.predict_generator(test_generator, test_iteration_count, verbose=1)
    predicted_labels = np.argmax(raw_predictions, axis=1)

    accuracy = accuracy_score(test_labels, predicted_labels)
    print("Accuracy: {}".format(accuracy))

    f3_score = fbeta_score(test_labels, predicted_labels, 3)
    print("F3 Score: {}".format(f3_score))

    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    print("Confusion Matrix: \n{}".format(conf_matrix))
