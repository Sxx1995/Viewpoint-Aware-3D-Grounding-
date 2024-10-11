import numpy as np
import math
from tqdm import tqdm
from random import sample
from matplotlib import pyplot as plt
import matplotlib
from copy import deepcopy
matplotlib.use('agg')
class semantic_relationship:
  def __init__(self):
    self.synonyms = {
       "a": ["a", "the", "this", "that"],
       "left": ["left to", "to the left of", "on the left side of"],
       "right": ["right to", "to the right of", "on the right side of"],
       "behind": ["behind"],
       "front": ["in front of"],
       "above": ["above", "on"],
       "below": ["below", "under"]
    }
  def get_rel(self, cls):
    idx = np.random.choice(np.arange(0, len(self.synonyms[cls])), size = 1)
    return self.synonyms[cls][idx[0]]

  def get_obj(self, name, first = False):
    idx = np.random.choice(np.arange(0, len(self.synonyms['a'])), size = 1)
    if first:
      return self.synonyms['a'][idx[0]] + ' ' + name
    else:
      s = np.random.rand(1)
      if s > 0.1:
        return 'it'
      else: 
        return self.synonyms['a'][idx[0]] + ' ' + name

  def get_sub(self, name, have_same, appear_object):
    idx = np.random.choice(np.arange(0, len(self.synonyms['a'])), size = 1)
    if have_same[0] in appear_object['name'] and not have_same[1] in appear_object['indx']:
      return 'the another ' + name
    elif have_same[0] in appear_object['name'] and have_same[1] in appear_object['indx']:
      return 'the ' + name
    else:
      return self.synonyms['a'][idx[0]] + ' ' + name

def box_iou(boxes2, boxes1, mode = '3d'):
    #boxes1 = torch.cat([torch.min(boxes1, dim = 2)[0], torch.max(boxes1, dim = 2)[0]], dim = -1)
    #boxes2 = torch.cat([torch.min(boxes2, dim = 2)[0], torch.max(boxes2, dim = 2)[0]], dim = -1)
    x_max_1 = boxes1[:, :, 3]
    y_max_1 = boxes1[:, :, 4]
    z_max_1 = boxes1[:, :, 5]
    x_min_1 = boxes1[:, :, 0]
    y_min_1 = boxes1[:, :, 1]
    z_min_1 = boxes1[:, :, 2]

    x_max_2 = boxes2[:, :, 3]
    y_max_2 = boxes2[:, :, 4]
    z_max_2 = boxes2[:, :, 5]
    x_min_2 = boxes2[:, :, 0]
    y_min_2 = boxes2[:, :, 1]
    z_min_2 = boxes2[:, :, 2]

    if mode == '3d':
        box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
        box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    else:
        box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
        box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    #lt = torch.max(boxes1[:, :, None, :3], boxes2[:, None, :, :3])  # [N,M,2]
    #rb = torch.min(boxes1[:, :, None, 3:], boxes2[:, None, :, 3:])  # [N,M,2]
    lt = np.maximum(boxes1[:, :, np.newaxis, :3], boxes2[:, np.newaxis, :, :3])  # [N,M,2]
    rb = np.minimum(boxes1[:, :, np.newaxis, 3:], boxes2[:, np.newaxis, :, 3:])  # [N,M,2]

    #wh = (rb - lt).clamp(min=0)  # [N,M,3]
    wh = np.clip(rb - lt, a_min=0, a_max = 10000000000)  # [N,M,3]

    if mode == '3d':
        inter_vol = wh[:, :, :, 0] * wh[:, :, :, 1] * wh[:, :, :, 2] # [N,M]
    else:
        inter_vol = wh[:, :, :, 0] * wh[:, :, :, 1] # [N,M]
    union = box_vol_1[:, :, None] + box_vol_2[:, None, :] - inter_vol

    iou = inter_vol / union
    iou_obj = inter_vol / box_vol_1[:, :, None]
    iou_sub = inter_vol / box_vol_2[:, None, :]
    return iou, iou_obj, iou_sub

def compute_all_relationships(target_object_ids, object_center, vec_matrix, rotated_matrix, bbox_list, _2d_ious, eps = 0.5, low_eps=0.2, _2d_eps=0.5):
  """
  Computes relationships between all pairs of objects in the scene.
  
  vec_matrix => [128, 128, 4] 
  rotated_matrix => [4, 4]
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  # Save all six axis-aligned directions in the scene struct
  scene_struct = {}
  scene_struct['directions'] = {}
  #scene_struct['directions']['behind'] = np.array([1, 0, 0])
  #scene_struct['directions']['front'] = np.array([-1, 0, 0])
  #scene_struct['directions']['left'] = np.array([0, -1, 0]) 
  #scene_struct['directions']['right'] = np.array([0, 1, 0])
  scene_struct['directions']['right'] = np.array([1, 0, 0])
  scene_struct['directions']['left'] = np.array([-1, 0, 0])
  scene_struct['directions']['front'] = np.array([0, -1, 0]) 
  scene_struct['directions']['behind'] = np.array([0, 1, 0])
  scene_struct['directions']['above'] = np.array([0, 0, 1])
  scene_struct['directions']['below'] = np.array([0, 0, -1])
  all_relationships_obj = {}
  all_relationships_obj_score = {}
  all_relationships_sub = {}
  all_relationships_sub_score = {}
  target_vectors_obj = vec_matrix[target_object_ids]
  _2d_ious = _2d_ious[0, target_object_ids]
  rotated_object_center = np.matmul(object_center, rotated_matrix[:3, :3])
  in_scope = rotated_object_center[:, 0] > 0
  for name, direction_vec in scene_struct['directions'].items():
    all_relationships_obj[name] = []
    all_relationships_sub[name] = []
    if name == 'below': #first ignore the above and below
      lower_rate = (bbox_list[target_object_ids, 2] - bbox_list[:, 2]) / (bbox_list[:, 5] - bbox_list[:, 2])
      upper = bbox_list[target_object_ids, 5] < bbox_list[:, 5]
      _2d_ious = _2d_ious > _2d_eps
      lower = lower_rate < low_eps
      is_below = _2d_ious * lower * upper * in_scope
      all_relationships_obj[name] = is_below
      all_relationships_obj_score[name] = is_below * 1 
    elif name == 'above':
      lower_rate = (bbox_list[target_object_ids, 2] - bbox_list[:, 2]) / (bbox_list[:, 5] - bbox_list[:, 2])
      upper = bbox_list[target_object_ids, 5] > bbox_list[:, 5]
      _2d_ious = _2d_ious > _2d_eps
      lower = lower_rate > low_eps
      is_upper = _2d_ious * lower * in_scope * upper
      all_relationships_obj[name] = is_upper
      all_relationships_obj_score[name] = is_upper * 1
    else:
      rotated_target_vectors_obj = np.matmul(target_vectors_obj, rotated_matrix)
      rotated_target_vectors_obj = rotated_target_vectors_obj[:, :3] 
      #rotated_target_vectors_obj = rotated_target_vectors_obj / ((rotated_target_vectors_obj**2).sum(-1)**0.5)[:, np.newaxis]
      dot = (rotated_target_vectors_obj * direction_vec[np.newaxis, :]).sum(-1) 
      cos_dot = dot / ((rotated_target_vectors_obj**2).sum(-1)**0.5)
      #if name == 'right':
      #  print(zzzzzzzzzz)
      noneoverlapping = _2d_ious < 0.1
      all_relationships_obj_score[name] = cos_dot * in_scope * noneoverlapping
      all_relationships_obj[name] = (cos_dot > eps) * (dot > 0.1) * in_scope * noneoverlapping
      all_relationships_sub[name] = (-cos_dot > eps) * (-dot > 0.1) * in_scope * noneoverlapping
  return all_relationships_obj, all_relationships_obj_score

def getting_contructed_sentence_with_random_center_and_direction(scene_id, \
                                                                 scene_construct, used_object_list, \
                                                                 k_sent_per_scene=100, \
                                                                 num_split = 10, num_angles = 36):
  """
  meta_data
  "scene_id": "scene0000_00",
  "object_id": "39",
  "object_name": "cabinet",
  "ann_id": "1",
  "description": "a white cabinet in the corner of the room. in the direction from the door and from the inside . it will be on the left, there is a small brown table on the left side of the cabinet and a smaller table on the right side of the cabinet",
  "token":[]
  """
  # get valid obejcts
  object_bbox = []
  object_name = []
  object_idx  = []
  idx = 0
  while True:
    try:
      if not scene_construct.get_object_instance_label(idx) in used_object_list:
        idx += 1
        continue
      object_idx.append(idx)
      object_bbox.append(scene_construct.get_object_bbox(idx))
      object_name.append(scene_construct.get_object_instance_label(idx))
    except:
      break
    idx += 1
  if len(object_bbox) < 2:
     return []

  # generate location grid
  pc = scene_construct.pc
  pc_min = pc.min(0)[np.newaxis, :]
  pc_max = pc.max(0)[np.newaxis, :]
  sampled_center = []
  for dh in range(num_split):
    for dw in range(num_split):
      sampled_center.append([dh / num_split, dw / num_split, 1])
  sampled_center = np.array(sampled_center)
  sampled_center = (pc_max - pc_min) * sampled_center + pc_min
  matrices = []

  # generate angles
  for i in range(num_angles):
    cos = np.cos(i * 2 * math.pi / num_angles) 
    sin = np.sin(i * 2 * math.pi / num_angles)
    rotate_matrix = np.array([[cos, -sin, 0, 0], \
                              [sin,  cos, 0, 0], \
                              [0,      0, 1, 0], \
                              [0,      0, 0, 1]])
    matrices.append(rotate_matrix)
  
  #center_object_idx = np.random.choice(list(range(len(object_bbox))), size = k_sent_per_scene)[0]
  meta_dicts = []
  for i in tqdm(range(k_sent_per_scene)):

    # get random location/angle/center_object
    angle_idx = np.random.choice(list(range(num_angles)), size = 1)[0]  
    center_idx = np.random.choice(list(range(int(num_split**2))), size = 1)[0]  
    center_object_idx = np.random.choice(list(range(len(object_bbox))), size = 1)[0]  

    # get relationships
    valid_relation, _ = sampled_relationships(object_bbox, object_name, object_idx, \
                                           sampled_center[center_idx], matrices[angle_idx], \
                                           center_object_idx)
    candidate_list = filter_samples(valid_relation)
    sentences = candidate_to_sentence(scene_id, i, \
                                      candidate_list, object_bbox, \
                                      sampled_center[center_idx], matrices[angle_idx])
    rotated_y_axis = np.matmul(np.array([[0, 1]]), np.linalg.inv(matrices[angle_idx][:2, :2]))

    assert abs(matrices[angle_idx][0][0] - rotated_y_axis[0][1]) < 1e-7
    assert abs(matrices[angle_idx][0][1] - rotated_y_axis[0][0]) <1e-7
    assert abs(matrices[angle_idx][1][0] - -rotated_y_axis[0][0]) <1e-7
    assert abs(matrices[angle_idx][1][1] - rotated_y_axis[0][1]) <1e-7
    # move all objects in candidate_list to the new lists
    idx_in_queue = []
    for j in candidate_list:
      idx_in_queue += [j[3]]
      idx_in_queue += [j[4]]
    idx_in_queue = list(set(idx_in_queue))
    filtered_bbox = []
    filtered_name = []
    filtered_idx = []
    filtered_center_object_idx = -1
    if idx_in_queue == []:
      continue
    for n, j in enumerate(idx_in_queue):
      filtered_bbox.append(deepcopy(object_bbox[j]))
      filtered_name.append(deepcopy(object_name[j]))
      filtered_idx.append(deepcopy(object_idx[j]))
      if j == center_object_idx:
        filtered_center_object_idx = n
    assert not filtered_center_object_idx == -1, "not find the selected_center_object"
    
    # find all valid angles
    valid_angle_vec = np.zeros((num_angles))
    max_angle_value = -1
    max_angle_index = 0
    for angle_idx, matrix in enumerate(matrices):
      valid_relation, score = sampled_relationships(filtered_bbox, filtered_name, filtered_idx, \
                                               sampled_center[center_idx], matrix, \
                                               filtered_center_object_idx)
      flag = True
      m = 1
      for j in candidate_list:
        if not j[:3] + [object_idx[j[3]], object_idx[j[4]]] in [x[:3] + [filtered_idx[x[3]], filtered_idx[x[4]]] for x in valid_relation[j[0]]]:
          flag = False
        else:
          idx = [x[:3] + [filtered_idx[x[3]], filtered_idx[x[4]]] for x in valid_relation[j[0]]].index(j[:3] + [object_idx[j[3]], object_idx[j[4]]])
          m = m * score[j[0]][idx]
      if flag:
        valid_angle_vec[angle_idx] += 1
        if m > max_angle_value: 
          max_angle_value = m
          max_angle_index = angle_idx

    #find all valid positions
    maps = np.zeros((num_split, num_split))
    for center_idx, center in enumerate(sampled_center):
      valid_relation, score = sampled_relationships(filtered_bbox, filtered_name, filtered_idx, \
                                             center, matrices[max_angle_index], \
                                             filtered_center_object_idx)
      flag = True
      for j in candidate_list:
        if not j[:3] + [object_idx[j[3]], object_idx[j[4]]] in [x[:3] + [filtered_idx[x[3]], filtered_idx[x[4]]] for x in valid_relation[j[0]]]:
          flag = False
        else:
          idx = [x[:3] + [filtered_idx[x[3]], filtered_idx[x[4]]] for x in valid_relation[j[0]]].index(j[:3] + [object_idx[j[3]], object_idx[j[4]]])
      if flag:
        maps[center_idx // num_split, center_idx % num_split] = round(m, 3)
        #maps[center_idx // num_split, center_idx % num_split] = max(maps[center_idx // num_split, center_idx % num_split], round(m, 3))
    meta_dict = save_to_dict(scene_id, i, \
                             center_object_idx, sentences[0], valid_angle_vec.tolist(), maps.tolist(), sentences[1], \
                             num_split, num_angles, object_idx, object_name)

    meta_dicts.append(meta_dict)
  return meta_dicts

def save_to_dict(scene_id, i, sampled_object, sentence, valid_angle_vec, valid_location, appear_objects, num_split, num_angles, object_idx, object_name):
  meta_dict = {}
  meta_dict["scene_id"] = scene_id
  meta_dict["object_id"] = object_idx[sampled_object]
  meta_dict["object_name"] = object_name[sampled_object]
  meta_dict["ann_id"] = 1000 + i
  meta_dict["description"] = sentence
  meta_dict["token"] = sentence.split(' ')
  meta_dict["num_angles"] = num_angles
  meta_dict["sampled_angle"]= valid_angle_vec
  meta_dict["sampled_center"] = valid_location
  appear_objects['indx'] = [object_idx[x] for x in appear_objects['indx']]
  meta_dict["appear_objects"] = appear_objects
  return meta_dict


def candidate_to_sentence(scene_id, sentence_num, candidate_list, object_list, sampled_center, rotate_matrix):
  relationships = semantic_relationship()
  sentences = []
  appear_object = {'name':[], 'indx':[]}
  idx_sent = 0
  if len(candidate_list) > 0:
    plot_into_fig(candidate_list[0][3], object_list, sampled_center, rotate_matrix, 'blue', candidate_list[0][1])
  for i in candidate_list:
    if len(appear_object['indx']) == 0:
      appear_object['name'].append(i[1])
      appear_object['indx'].append(int(i[3]))
    sentences.append(relationships.get_obj(i[1], idx_sent == 0) + ' is ' + \
                     relationships.get_rel(i[0]) + ' ' + \
                     relationships.get_sub(i[2], [i[2], i[4]], appear_object) + ' .'\
                    )
    appear_object['name'].append(i[2])
    appear_object['indx'].append(int(i[4]))
    plot_into_fig(i[4], object_list, sampled_center, rotate_matrix, 'green', i[2]+str(i[4]))
    idx_sent += 1
  sentences = ' '.join(sentences)
  plt.savefig('imgs/' + scene_id + str(sentence_num) + '.jpg')
  plt.cla()
  return sentences, appear_object, candidate_list

def filter_samples(valid_relation, number_of_selection_from_each_relation = 1, number_of_relation = 4):
  candidate_list = []
  for i in valid_relation:
    if number_of_selection_from_each_relation > len(valid_relation[i]):
      candidate_list += valid_relation[i]
    else:
      candidate_list += sample(valid_relation[i], \
                               number_of_selection_from_each_relation)
  if number_of_relation < len(candidate_list):
    candidate_list = sample(candidate_list, \
                            number_of_relation)
  return candidate_list


def sampled_relationships(object_bbox, object_name, object_idx, \
                          sampled_center, rotate_matrix, obj_idx, \
                          number_of_selection_from_each_relation = 1, number_of_relation = 4):
  object_bbox = np.array(object_bbox)
  object_center = (object_bbox[:, 3:] + object_bbox[:, :3])/2 - sampled_center
  ious, ious_obj, ious_sub = box_iou(object_bbox[np.newaxis, :, :], \
                                     object_bbox[np.newaxis, :, :], \
                                     mode = '2d')

  object_distance = object_center[:, np.newaxis, :] - \
                    object_center[np.newaxis, :, :]

  object_distance = np.concatenate([object_distance, \
                                    np.ones((object_distance.shape[0], object_distance.shape[0], 1))], \
                                   axis = 2)


  all_relationships, scores = compute_all_relationships(obj_idx, \
                                                object_center, object_distance, rotate_matrix, object_bbox, ious_obj)
  valid_relation = {}
  valid_relation_score = {}
  for i in all_relationships:
    valid_relation[i] = []
    valid_relation_score[i] = []
    for sub_idx, j in enumerate(all_relationships[i]):
      if j:
         valid_relation[i].append([i,                                                       #relation 
                                   #scores[i][sub_idx]
                                   object_name[obj_idx], object_name[sub_idx],              #name of objects 
                                   #'obj_idx': [object_idx[obj_idx], object_idx[sub_idx]],                #indx of objects
                                   obj_idx, sub_idx])                                       #indx of objects
         valid_relation_score[i].append(scores[i][sub_idx])
  return valid_relation, valid_relation_score

def plot_into_fig(idx, object_list, sampled_center, rotated_matrix, color, text):
  x_min = object_list[idx][0]
  y_min = object_list[idx][1]
  x_max = object_list[idx][3]
  y_max = object_list[idx][4]
  bbox = np.array([[x_min, y_min],
                      [x_min, y_max],
                      [x_max, y_max],
                      [x_max, y_min],
                      [x_min, y_min]])
  bbox = bbox - sampled_center[np.newaxis, :2]
  rotated_bbox = np.matmul(bbox, rotated_matrix[:2, :2])
  plt.plot(rotated_bbox[:, 0], rotated_bbox[:, 1], '.-%s'%(color[:1]))
  plt.text(rotated_bbox.max(0)[0], rotated_bbox.max(0)[1], text, fontsize=15, verticalalignment="top", horizontalalignment="right")

