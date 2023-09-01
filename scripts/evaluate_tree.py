import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from dvclive import Live
from sklearn import metrics
from constants import EVAL_PATH, cols


def evaluate(model, X, y, split, live):
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    if not live.summary:
        live.summary = {'accuracy': {}, 'avg_prec': {}, 'roc_auc': {}}
    live.summary['accuracy'][split] = metrics.accuracy_score(y, predictions_by_class.argmax(-1))
    live.summary['avg_prec'][split] = metrics.average_precision_score(y, predictions)
    live.summary['roc_auc'][split]  = metrics.roc_auc_score(y, predictions)

    live.log_sklearn_plot('precision_recall', y, predictions, name=f'prc/{split}', drop_intermediate=True)
    live.log_sklearn_plot('roc', y, predictions, name=f'roc/{split}')
    live.log_sklearn_plot('confusion_matrix', y, predictions_by_class.argmax(-1), name=f'cm/{split}')

    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel('Mean decrease in impurity')

    importances = model.feature_importances_
    features_importances = pd.Series(importances, index=cols[:-1]).nlargest(n=30)
    features_importances.plot.barh(ax=axes)

    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    image_pil = Image.open(buffer)
    live.log_image(name=f'features_importance_{split}.png', val=image_pil)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate_tree.py features-dir model-path')
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
        clf_tree = pickle.load(fd)

    # Evaluate train and test datasets
    eval_path = os.path.join(EVAL_PATH, 'live_tree')
    live = Live(eval_path, dvcyaml=False)

    evaluate(clf_tree, X_train, y_train, 'train', live)
    evaluate(clf_tree, X_test, y_test, 'test', live)

    live.make_summary()
