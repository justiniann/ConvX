from preprocessing.Dataset import *
import os
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import applications
from keras import optimizers


def train_model(total_images=None, epochs=5):
    train_imgs, train_targets, validation_imgs, validation_targets, test_imgs, test_targets = split_that_shit(size=total_images)
    img_vectors_train = vectorize_images(train_imgs).astype('float32')/255
    img_vectors_validation = vectorize_images(validation_imgs).astype('float32')/255
    img_vectors_test = vectorize_images(test_imgs).astype('float32')/255

    model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(512, 512, 3))

    for layer in model.layers:
        layer.trainable = False

    # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model
    model_final = Model(input=model.input, output=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])

    checkpointer = ModelCheckpoint(filepath='..{0}..{0}saved_models{0}resnet50_best.hdf5'.format(os.path.sep),
                                   verbose=0, save_best_only=True)

    model_final.fit(img_vectors_train, train_targets, validation_data=(img_vectors_validation, validation_targets),
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=0)

    model_final.load_weights('..{0}..{0}saved_models{0}resnet50_best.hdf5'.format(os.path.sep))

    # get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(model_final.predict(np.expand_dims(img, axis=0))) for img in img_vectors_test]

    # report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(test_targets)
    print('Test accuracy: %.4f%%' % test_accuracy)
    print("Training Done!")


if __name__ == '__main__':
    train_model(total_images=1000, epochs=5)
