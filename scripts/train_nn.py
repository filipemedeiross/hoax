import os
import sys
import yaml
import pickle
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train_nn.py dir-features dir-model')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train_nn']
    lr = params['lr']
    epochs = params['epochs']
    bs = params['bs']

    input = sys.argv[1]
    output = os.path.join(sys.argv[2], 'clf_nn.pkl')

    with open(os.path.join(input, 'train.pkl'), 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    print('Training the Neural Network...')
    DS = X_train.shape[0]
    d  = X_train.shape[1]
    n  = int((DS-10) / (10*(d+2)))

    clf_nn = Sequential()
    clf_nn.add(Dense(n, input_dim=d, kernel_initializer='normal', activation='tanh'))
    clf_nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    adam = optimizers.Adam(learning_rate=lr)
    clf_nn.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    clf_nn.fit(X_train, y_train, epochs=epochs, batch_size=bs, verbose=0)

    print(f'Saving the model in {output}.')
    with open(output, 'wb') as fd:
        pickle.dump(clf_nn, fd)
