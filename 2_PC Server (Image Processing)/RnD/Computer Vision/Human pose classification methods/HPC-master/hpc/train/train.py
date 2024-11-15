import argparse

import hpc.consts as c
import hpc.core.model as model
import numpy as np
# os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
import tensorflow as tf
from matplotlib import pyplot as plt


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_ds", help="Name or path to your dataset relative to /dataF/datasets.")
    parser.add_argument("-v", "--validation_ds", help="Path to validation dataset relative to /dataF/datasets.")
    parser.add_argument("-m", "--model_name", help="Name of model you want to continue training. If none, new model will be created.")
    parser.add_argument("-o", "--output_model", help="Name of output model. If none, model won't be saved.")
    parser.add_argument("-z", "--no_z", action="store_true")
    return parser.parse_known_args()[0]


def readDataset(dsName):
    with np.load("dataF/datasets/" + dsName, allow_pickle=True) as data:
        img = data['images']
        lab = data['labels']
    print("Dataset loaded.")
    if args.no_z:
        img[:, :, :, 0] = np.zeros(img.shape[:3])
    return img, lab


def shuffleAndSplit(ds, datasetSize, trainSizeFactor):
    if not 0 < trainSizeFactor <= 1:
        raise ValueError("Train size factor must be in <0, 1>")
    ds = ds.shuffle(datasetSize)
    return ds.take(int(trainSizeFactor * datasetSize)), ds.skip(int(trainSizeFactor * datasetSize))


def getModel():
    try:
        mod = tf.keras.models.load_model('dataF/yolo_models_utils/' + args.model_name)
        print("Model " + args.model_name + " loaded.")
    except Exception:
        print("Creating new model.")
        mod = model.getModel()
    print()
    print(mod.summary())
    return mod


def saveModel():
    if args.output_model is not None:
        m.save('dataF/yolo_models_utils/' + args.output_model)
        print("Model saved to /dataF/yolo_models_utils/" + args.output_model)
    else:
        print("Model not saved.")


def showPlots(hist, save=None):
    # accuracy = plt.figure( 0 )
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save is not None:
        plt.savefig(f"../thesis/figuresTemp/{args.output_model}{save}Acc{c.epochs}.png")
    plt.clf()
    # loss = plt.figure( 1 )
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save is not None:
        plt.savefig(f"../thesis/figuresTemp/{args.output_model}{save}Loss{c.epochs}.png")
    plt.clf()
    # plt.show()


if __name__ == '__main__':
    args = parseArgs()
    images, labels = readDataset(args.train_ds)
    images = images / 255.0
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if args.validation_ds is not None:
        train, _ = shuffleAndSplit(dataset, len(images), 1.0)
        images, labels = readDataset(args.validation_ds)
        images = images / 255.0
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        validation, _ = shuffleAndSplit(dataset, len(images), 1.0)
    else:
        train, validation = shuffleAndSplit(dataset, len(images), 0.8)

    train = train.batch(c.batchSize)
    validation = validation.batch(c.batchSize)

    output_model = args.output_model
    m = getModel()
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath='dataF/yolo_models_utils/' + output_model,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    history = m.fit(train, epochs=c.epochs, batch_size=c.batchSize, validation_data=validation, callbacks=[mc])

    acc = max(history.history['val_accuracy'])
    idx = history.history['val_accuracy'].index(acc)
    loss = history.history['val_loss'][idx]
    showPlots(history.history)
    file = open("info.txt", "a")
    file.write(f"\n{output_model}, accuracy = {acc}, loss = {loss}, batch size = {c.batchSize}, after epoch {idx + 1}")
    file.close()
