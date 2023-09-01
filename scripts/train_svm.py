import os
import sys
import yaml
import pickle
from sklearn.svm import SVC


if __name__ == '__main__':    
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train_svm.py dir-features dir-model')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train_svm']
    C = params['C']
    gamma = params['gamma']

    input = sys.argv[1]
    output = os.path.join(sys.argv[2], 'clf_svm.pkl')

    with open(os.path.join(input, 'train.pkl'), 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    print('Training the SVM...')
    clf_svm = SVC(C=C, gamma=gamma)
    clf_svm.fit(X_train, y_train)

    print(f'Saving the model in {output}.')
    with open(output, 'wb') as fd:
        pickle.dump(clf_svm, fd)
