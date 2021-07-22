import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import models, regularizers, layers, optimizers, metrics, losses, initializers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

def preproccesing (data, split, normalization='', shuffle=True):
    if shuffle:
        np.random.shuffle(data)

    N = data.shape[0]
    xtr = data[:int(split*N),:-1]
    xte = data[int(split*N):,:-1]
    ytr = data[:int(split*N):, -1:]
    yte = data[int(split*N):, -1:]

    if normalization == 'min_max':
        scaler = MinMaxScaler(feature_range=(-1,1.))
        xtr = scaler.fit_transform(xtr)
        xte = scaler.fit_transform(xte)
    elif normalization == 'standard':
        standarizer = StandardScaler()
        xtr = standarizer.fit_transform(xtr)
        xte = standarizer.fit_transform(xte)
    elif normalization != '':
        raise RuntimeError('normalization argument not valid')

    return xtr, ytr, xte, yte

def plot_history(x, hist, title):
    plt.figure()
    plt.title('Loss ' + title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.plot(x, hist['loss'], label='Loss')
    plt.plot(x, hist['val_loss'], label='Loss validation data')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('images/loss_' + title + '.jpg', dpi=300)
    plt.close()

    plt.figure()
    plt.title('Accuracy ' + title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(x, hist['accuracy'], label='Accuracy')
    plt.plot(x, hist['val_accuracy'], label='Accuracy validation data')
    # M = hist['accuracy'].max()
    # iM = hist['accuracy'].argmax()
    # plt.scatter(iM, M, label=r'Máx. acc {:.3f}'.format(M))
    plt.legend(loc='lower right', fontsize=10, fancybox=True, shadow=True)
    plt.grid()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('images/acc_' + title + '.jpg', dpi=300)
    plt.close()

def get_model (input_shape):
    activation = 'relu'
    dropout = 0.3 # 0.3
    initializer = 'glorot_uniform'

    # begin model
    model = models.Sequential(name='NetBA')

    model.add(layers.Dense(
        input_shape=input_shape,
        units=128,
        use_bias=True,
        kernel_initializer = initializer,
        activation=activation,
        ))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(
        units=64,
        use_bias=True,
      	kernel_initializer = initializer,
        activation=activation,
        ))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(
        units=32,
        use_bias=True,
        kernel_initializer = initializer,
        activation=activation,
        ))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(
        units=8,
        use_bias=True,
        kernel_initializer = initializer,
        activation=activation,
        ))

    model.add(layers.Dense(
        units=1,
        use_bias=False,
        activation= 'sigmoid'
        ))

    opt = optimizers.Adam(learning_rate=0.005) # 0.005
    metric = metrics.BinaryAccuracy(name='accuracy', threshold=0.5)
    loss = losses.BinaryCrossentropy(from_logits=False)

    model.compile(opt, loss=loss, metrics=metric)
    #model.summary()
    # end model

    return model

def main():
    data = pd.read_csv('files/2016-17.csv').to_numpy()
    xtr, ytr, xte, yte = preproccesing(data, .7, shuffle=True) #.7

    # model = get_model((xtr.shape[1],))

    # history=model.fit(xtr, ytr, batch_size=150, epochs=128, # bs 128
    #                   verbose=2, validation_split=0.1, shuffle=False)

    # hist = pd.DataFrame(history.history)
    # plot_history(history.epoch, hist, '2012-19-relu')

    # prediction = np.round(model.predict(xte, verbose=1)) # predicciones
    # print(np.mean(prediction==yte))
    #

    # cross validation
    k_folds = 5
    epochs = 200
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    skf.get_n_splits(xtr, ytr)

    history = []
    acc_pred = 0
    for train_index, test_index in skf.split(xtr, ytr):
        model = get_model((xtr.shape[1],))
        x, x_val = xtr[train_index], xtr[test_index]
        y, y_val = ytr[train_index], ytr[test_index]

        history.append(model.fit(x, y, batch_size=256, epochs=epochs, # 128, 256
                          validation_data=(x_val,y_val), verbose=2, shuffle=True))

        prediction = np.round(model.predict(xte, verbose=1)) # predicciones
        print(np.mean(prediction==yte))
        acc_pred += np.mean(prediction==yte)

    print('Mean prediction accuracy: {:.4f}'.format(acc_pred / k_folds))

    # results post-proccesing
    prom_loss = np.zeros((epochs,), dtype=np.float32)
    prom_loss_val = np.zeros((epochs,), dtype=np.float32)
    prom_acc = np.zeros((epochs,), dtype=np.float32)
    prom_acc_val = np.zeros((epochs,), dtype=np.float32)

    for i in range(k_folds):
        x = pd.DataFrame(history[i].history)
        prom_loss += x['loss']
        prom_loss_val += x['val_loss']
        prom_acc += x['accuracy']
        prom_acc_val += x['val_accuracy']

    prom_loss /= k_folds
    prom_loss_val /= k_folds
    prom_acc /= k_folds
    prom_acc_val /= k_folds
    hist = pd.concat([prom_loss, prom_acc, prom_loss_val, prom_acc_val], axis=1)
    plot_history(history[0].epoch, hist, '2016 - 2017')


    return 0

if __name__ == '__main__':
    main()
