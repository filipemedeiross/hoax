import os
import sys
import yaml
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':    
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train_tree.py dir-features dir-model')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['train_tree']
    cv = params['cv']

    input = sys.argv[1]
    output = os.path.join(sys.argv[2], 'clf_tree.pkl')

    with open(os.path.join(input, 'train.pkl'), 'rb') as fd:
        X_train, y_train = pickle.load(fd)

    print('Training the Decision Tree Classifier...')
    clf_tree = DecisionTreeClassifier()

    ccp_alphas = clf_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    param_grid_tree = {'ccp_alpha' : ccp_alphas[ccp_alphas > 0]}

    CV_clf_tree = GridSearchCV(estimator=clf_tree, param_grid=param_grid_tree, cv=cv)
    CV_clf_tree.fit(X_train, y_train)

    print(f'Saving the model in {output}.')
    with open(output, 'wb') as fd:
        pickle.dump(CV_clf_tree.best_estimator_, fd)
