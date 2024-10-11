from six.moves import cPickle
import json
import numpy as np
from prompt_processing import box_iou, compute_all_relationships, semantic_relationship, getting_sentence_with_rotated_matrix, \
                              getting_contructed_sentence_with_random_center_and_direction
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()
q = unpickle_data('DATA_ROOT/train_v3scans.pkl')
used_object_list = json.load(open('DATA_ROOT/scanrefer_object_names.json', 'r'))
with open('DATA_ROOT/scanrefer/ScanRefer_filtered_train.txt', 'r') as f:
    scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]

q = list(q)[0]
syn_meta_data = []
for i in scan_ids:
    tmp = getting_contructed_sentence_with_random_center_and_direction(i, q[i], used_object_list, k_sent_per_scene = 50)
    syn_meta_data += tmp
tmp = []
for i in syn_meta_data:
    if not i['description'] == '':
        tmp.append(i) 
json.dump(syn_meta_data, open('meta_data_process/scanrefer/ScanRefer_filtered_train.json', 'w'))
