import os
import sys
import pickle
import numpy as np
from dvclive import Live
from sklearn import metrics
from constants import EVAL_PATH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def evaluate(model, X, y, split, live):
    predictions = model.predict(X, verbose=0).squeeze().astype('float64')
    
    if not live.summary:
        live.summary = {'accuracy': {}, 'avg_prec': {}, 'roc_auc': {}}
    live.summary['accuracy'][split] = metrics.accuracy_score(y, np.round(predictions))
    live.summary['avg_prec'][split] = metrics.average_precision_score(y, predictions)
    live.summary['roc_auc'][split]  = metrics.roc_auc_score(y, predictions)

    live.log_sklearn_plot('roc', y, predictions, name=f'roc/{split}')
    live.log_sklearn_plot('confusion_matrix', y, np.round(predictions), name=f'cm/{split}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate_nn.py features-dir model-path')
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
        clf_nn = pickle.load(fd)

    # Evaluate train and test datasets
    eval_path = os.path.join(EVAL_PATH, 'live_nn')
    live = Live(eval_path, dvcyaml=False)

    evaluate(clf_nn, X_train, y_train, 'train', live)
    evaluate(clf_nn, X_test, y_test, 'test', live)

    live.make_summary()
