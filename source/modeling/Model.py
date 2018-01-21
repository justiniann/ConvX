from preprocessing.Dataset import *
import os
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import applications
from keras import optimizers
import time
from keras import backend as K
from sklearn.metrics import fbeta_score


LAYERS_TO_FREEZE = 172


def fbeta(y_true, y_pred, beta=2, threshold_shift=0):
    """Calculate fbeta score for Keras metrics.
    from https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras
    """
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


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=[fbeta]
    )
    return model


def setup_to_finetune(model):
    for layer in model.layers[:LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(
        optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=[fbeta]
    )
    return model


def build_model_from_scratch(output_nodes):
    model = Sequential()
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='same', activation='relu', input_shape=(224, 224, 3))
    model.add(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')
    model.add(pool1)
    conv2 = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')
    model.add(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')
    model.add(pool2)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='relu')
    model.add(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')
    model.add(pool3)
    gap1 = GlobalAveragePooling2D()
    model.add(gap1)
    model.add(Dense(output_nodes, activation="softmax"))
    return model


def build_batch_generator(X, y, batch_size, number_of_batches):
    counter = 0
    number_of_batches = np.ceil(X.shape[0] / batch_size)
    while True:
        idx_start = batch_size * counter
        idx_end = batch_size * (counter + 1)
        if idx_end > len(X):
            idx_end = len(X)
        x_batch = vectorize_images(X[idx_start:idx_end]).astype('float32')/255
        y_batch = np_utils.to_categorical(y[idx_start:idx_end], 2)
        counter += 1
        yield x_batch, y_batch
        if counter == number_of_batches-1:
            counter = 0


def train_model(total_images=None, batch_size=32, epochs=5):
    print("Starting image vectorization process...")
    start_time = time.time()
    train_X, train_y, validation_X, validation_y, test_X, test_y = split_that_shit(size=total_images)

    steps_per_epoch_fit = np.ceil(len(train_X) / batch_size)
    steps_per_epoch_val = np.ceil(len(validation_X) / batch_size)

    training_generator = build_batch_generator(train_X, train_y, batch_size, steps_per_epoch_fit)
    validation_generator = build_batch_generator(validation_X, validation_y, batch_size, steps_per_epoch_val)

    print("Done. Total time: {} sec".format(time.time() - start_time))

    print("Beginning fitting network...")
    start_time = time.time()
    checkpointer = ModelCheckpoint(
        filepath='..{0}..{0}saved_models{0}resnet50_best.hdf5'.format(os.path.sep),
        verbose=0,
        save_best_only=True
    )
    base_model = applications.InceptionV3(weights='imagenet', include_top=False)
    model = add_new_last_layer(base_model, 2)

    # transfer learning
    model = setup_to_transfer_learn(model, base_model)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch_fit,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_val,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer]
    )

    # fine-tuning
    model = setup_to_finetune(model)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch_fit,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_val,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointer]
    )

    model.load_weights('..{0}..{0}saved_models{0}resnet50_best.hdf5'.format(os.path.sep))
    print("Done. Total time: {} sec".format(time.time() - start_time))

    # get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(model.predict(
        np.expand_dims(vectorize_img(img), axis=0))
    ) for img in test_X]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_y, axis=1)) / len(test_y)
    print('Test accuracy: %.4f%%' % test_accuracy)
    print("Training Done!")


if __name__ == '__main__':
    train_model(epochs=5, total_images=100)
