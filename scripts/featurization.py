import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from tools import *
from constants import cols
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def process_dataframe(dataframe):
    dataframe['validado'] = dataframe['validado'].map({'N' : 0, 'S' : 1})
    dataframe['ocorrencia_oleo'] = dataframe['ocorrencia_oleo'].map({'N' : 0, 'S' : 1})
    dataframe['informacao_responsavel'] = dataframe['informacao_responsavel'].map({'N' : 0, 'T' : 1})
    dataframe['informacao_geografica'] = dataframe['informacao_geografica'].map({np.nan : 0, 'S' : 1})
    dataframe['plano_emergencia_acionado'] = dataframe['plano_emergencia_acionado'].map({'N' : 0, 'S' : 1})
    dataframe['iniciados_outras_providencias'] = dataframe['iniciados_outras_providencias'].map({'N' : 0, 'S' : 1})

    dataframe.dropna(subset=['periodo_ocorrencia'], inplace=True)
    le_period = LabelEncoder()
    le_period.fit(dataframe['periodo_ocorrencia'])
    dataframe['periodo_ocorrencia'] = le_period.transform(dataframe['periodo_ocorrencia'])  

    dataframe.dropna(subset=['uf'], inplace=True)
    le_uf = LabelEncoder()
    le_uf.fit(dataframe['uf'])
    dataframe['uf'] = le_uf.transform(dataframe['uf'])

    dataframe['des_ocorrencia'] = dataframe['des_ocorrencia'].apply(is_str)

    dataframe.dropna(subset=['origem'], inplace=True)
    dataframe['origem'] = dataframe['origem'].apply(first_str)
    le_origin = LabelEncoder()
    le_origin.fit(dataframe['origem'])
    dataframe['origem'] = le_origin.transform(dataframe['origem'])

    dataframe['institiuicoes_atuando_local'] = dataframe['institiuicoes_atuando_local'].apply(count_str)
    dataframe.rename(columns={'institiuicoes_atuando_local' : 'quant_instituicoes_atuando_local'}, inplace=True)

    dataframe['tipos_fontes_informacoes'] = dataframe['tipos_fontes_informacoes'].apply(count_str)
    dataframe.rename(columns={'tipos_fontes_informacoes' : 'quant_fontes_informacoes'}, inplace=True)

    dataframe['tipos_danos_identificados'] = dataframe['tipos_danos_identificados'].apply(count_str)
    dataframe.rename(columns={'tipos_danos_identificados' : 'quant_tipos_danos_identificados'}, inplace=True)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython featurization.py data-path features-dir-path')
        sys.exit(1)

    params = yaml.safe_load(open('params.yaml'))['featurize']
    seed = params['seed']
    split_val = params['split_val']
    split_test = params['split_test']

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
    
    data = sys.argv[1]
    train_output = os.path.join(sys.argv[2], 'train.pkl')
    val_output = os.path.join(sys.argv[2], 'val.pkl')
    test_output = os.path.join(sys.argv[2], 'test.pkl')

    notifications = pd.read_csv(data, low_memory=False, usecols=cols)[cols]
    process_dataframe(notifications)

    X = notifications.drop(['validado'], axis=1).to_numpy()
    y = notifications['validado'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_val, random_state=seed)

    with open(train_output, 'wb') as fd:
        pickle.dump((X_train, y_train), fd)

    with open(val_output, 'wb') as fd:
        pickle.dump((X_val, y_val), fd)

    with open(test_output, 'wb') as fd:
        pickle.dump((X_test, y_test), fd)
