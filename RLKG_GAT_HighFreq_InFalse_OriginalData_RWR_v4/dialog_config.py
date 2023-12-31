import ast
import pickle

import torch
import os
# from utils.utils import *

def save_pickle(a, path):
    with open(path, 'wb') as f:
        pickle.dump(a, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = {}
    with open(path, 'r', encoding='utf-8') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set


def text_to_list(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """

    slot_set = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            slot_set.append(line.strip('\n').strip('\r'))
    return slot_set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
diagnosis_re = os.path.join(father_path, 'output', 'train_re.txt')
test_re = os.path.join(father_path, 'output', 'test_re.txt')
buffer_re = os.path.join(father_path, 'output', 'buffer.txt')
action_rest_re = os.path.join(father_path, 'output', 'action_rest.txt')
action_re = True
dis_sym_num_to_graph_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'dise_sym_num_high_dict(≥3).txt')
# dis_sym_num_to_graph_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'dise_sym_num_dict_infalse.txt')
# dis_sym_num_to_graph_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'dise_sym_num_high_dict_infalse(≥3).txt')
disease_mask = torch.ones(12)
symptoms_mask = torch.ones(118)
include_sym_is_empty = False
# print(diagnosis_re)
################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

################################################################################
#  Slot Values
################################################################################
TRUE = 1  # 有该症状
FALSE = -1  # 无该症状
NOT_SURE = 0.6  # 不确定是否有
NOT_MENTION = 0.3  # 未提问

################################################################################
#  Diagnosis
################################################################################
NO_DECIDE = 0
NO_MATCH = "no match"
NO_MATCH_BY_RATE = "no match by rate"

# dis_sym_num_to_graph
with open(dis_sym_num_to_graph_path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    dis_sym_num_to_graph = ast.literal_eval(content[0])

# gmd
current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path))
disease_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'gmd_disease.txt')
disease = text_to_list(disease_path)
# print(disease)
symptoms_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'gmd_symptom.txt')
symptoms = text_to_list(symptoms_path)
# print(symptoms)
sym_high_path = os.path.join(father_path, 'KGData', 'dataset_gmd', 'req_dise_sym_dict.p')
req_dise_sym = load_pickle(sym_high_path)
# print(req_dise_sym)
sym_high = []
for d in list(req_dise_sym.keys()):
    for s in req_dise_sym[d]:
        if s not in sym_high:
            sym_high.append(s)
# print(sym_high)
# print(len(sym_high))
sys_inform_slots_values = ['食管炎', '肠炎', '哮喘', '冠心病', '肺炎', '鼻炎', '甲状腺炎', '脑外伤', '皮炎', '外耳炎', '结膜炎', '乳腺炎']
sys_request_slots = symptoms
sys_request_slots_highfreq = sym_high
# print(sys_request_slots)
# print(len(sys_request_slots))
# print(sys_request_slots_highfreq)
# print(len(sys_request_slots_highfreq))