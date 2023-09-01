import os
import sys
import pickle
from dvclive import Live
from sklearn import metrics
from constants import EVAL_PATH


def evaluate(model, X, y, split, live):
    predictions = model.predict(X)

    if not live.summary:
        live.summary = {'accuracy': {}}
    live.summary['accuracy'][split] = metrics.accuracy_score(y, predictions)

    live.log_sklearn_plot('confusion_matrix', y, predictions, name=f'cm/{split}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate_svm.py features-dir model-path')
        sys.exit(1)

    # Load data and model
    train_path = os.path.join(sys.argv[1], 'train.pkl')
    test_path  = os.path.join(sys.argv[1], 'test.pkl')
    model_path = sys.argv[2]

    with open(train_path, 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    with open(test_path, 'rb') as fd:
        X_test, y_test = pickle.load(fd)
    
    with open(model_path, 'rb') as fd:
        clf_svm = pickle.load(fd)

    # Evaluate train and test datasets
    eval_path = os.path.join(EVAL_PATH, 'live_svm')
    live = Live(eval_path, dvcyaml=False)

    evaluate(clf_svm, X_train, y_train, 'train', live)
    evaluate(clf_svm, X_test, y_test, 'test', live)

    live.make_summary()
