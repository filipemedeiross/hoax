import numpy as np
import pandas as pd


def is_str(data):
    return 1 if pd.notna(data) and data != 'Não há descrição' else 0

def first_str(data):
    return data.split(';')[0]

def count_str(data):
    return len(data.split(';')) if pd.notna(data) else 0
