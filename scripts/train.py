import os
import sys
import yaml
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense


if __name__ == '__main__':    
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train.py dir-features dir-model')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train']
    cv = params['cv']
    lr = params['lr']
    epochs = params['epochs']
    bs = params['bs']

    input = sys.argv[1]
    output = os.path.join(sys.argv[2], 'model.pkl')

    with open(os.path.join(input, 'train.pkl'), 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    with open(os.path.join(input, 'val.pkl'), 'rb') as fd:
        X_val, y_val = pickle.load(fd)

    print('Training the Decision Tree Classifier...')
    clf_tree = DecisionTreeClassifier()

    ccp_alphas = clf_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    param_grid_tree = {'ccp_alpha' : ccp_alphas[ccp_alphas > 0]}

    CV_clf_tree = GridSearchCV(estimator=clf_tree, param_grid=param_grid_tree, cv=cv)
    CV_clf_tree.fit(X_train, y_train)

    print('Training the SVM...')
    param_grid_svm = {'C' : [1, 2, 5, 10],
                      'gamma' : [0.1, 0.01, 1, 5]}

    CV_clf_svm = GridSearchCV(estimator=SVC(), param_grid=param_grid_svm, cv=cv)
    CV_clf_svm.fit(X_train, y_train)

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

    print('Evaluating the models -> ', end='')
    names  = ['DecisionTreeClassifier', 'SVM', 'NN']
    models = [CV_clf_tree.best_estimator_, CV_clf_svm.best_estimator_, clf_nn]
    best_clf = np.argmin([1 - models[0].score(X_val, y_val),
                          1 - models[1].score(X_val, y_val),
                          1 - accuracy_score(y_val, np.round(models[2].predict(X_val, verbose=0)))])

    print('The best model was', names[best_clf])
    with open(output, 'wb') as fd:
        pickle.dump(models[best_clf], fd)
