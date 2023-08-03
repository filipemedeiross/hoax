import os
import sys
import pickle
from dvclive import Live
from sklearn import metrics
from constants import EVAL_PATH


def evaluate(model, X, y, split, live):
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    if not live.summary:
        live.summary = {'avg_prec': {}, 'roc_auc': {}}
    live.summary['avg_prec'][split] = metrics.average_precision_score(y, predictions)
    live.summary['roc_auc'][split]  = metrics.roc_auc_score(y, predictions)

    live.log_sklearn_plot('roc', y, predictions, name=f'roc/{split}')
    live.log_sklearn_plot('confusion_matrix', y, predictions_by_class.argmax(-1), name=f'cm/{split}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py dir-features dir-model')
        sys.exit(1)

    # Load data and model
    train_file = os.path.join(sys.argv[1], 'train.pkl')
    test_file  = os.path.join(sys.argv[1], 'test.pkl')
    model_file = os.path.join(sys.argv[2], 'model.pkl')

    with open(train_file, 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    with open(test_file, 'rb') as fd:
        X_test, y_test = pickle.load(fd)
    
    with open(model_file, 'rb') as fd:
        model = pickle.load(fd)

    # Evaluate train and test datasets
    live = Live(os.path.join(EVAL_PATH, 'live'), dvcyaml=False)
    evaluate(model, X_train, y_train, 'train', live)
    evaluate(model, X_test, y_test, 'test', live)
    live.make_summary()
