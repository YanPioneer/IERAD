import os
import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_data():
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    to_path = os.path.join(father_path, 'GAT-Data', 'data')
    data = load_pickle(os.path.join(to_path, 'disease_gmd.p'))
    # print(data)
    return data

#
# data_ = load_data()
# print(len(data_['train']))
# print(data_.keys())
# print(data_['train'][0])
