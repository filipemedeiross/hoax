import os
import sys
from pandas import read_csv
from constants import cols


def load_data(url, filepath):
    try:
        print('Loading the files from the url...')
        notifications = read_csv(url, low_memory=False, usecols=cols)[cols]
    except:
        sys.stderr.write('Could not load data from url!')
        sys.exit(1)

    notifications.to_csv(filepath)
    print(f'File saved in {os.path.join(os.getcwd(), filepath)}.')


if __name__ == '__main__':
    url = 'http://siscom.ibama.gov.br/geoserver/publica/' + \
          'ows?service=WFS&version=1.0.0&request=GetFeature&typeName=publica:' + \
          'adm_comunicacidente_p&outputFormat=csv'
    filepath = os.path.join('data', 'acidenteambiental.csv')

    load_data(url, filepath)
