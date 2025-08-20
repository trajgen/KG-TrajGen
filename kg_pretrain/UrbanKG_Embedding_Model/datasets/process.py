"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle

import numpy as np


DATA_PATH = "../data"
def get_idx(path):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)

    #内容形式为 entityid to id(0，1，2，3...)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}  #集合确保里面的元素是唯一的
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    return ent2idx, rel2idx


'''
            将训练集、验证集和测试集的原始的实体ID映射成新ID
            entityid to id(0，1，2，3...)
'''
def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    #读取train valid test
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)  #逆三元组？
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path, dataset_name):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    ent2idx, rel2idx = get_idx(dataset_path)

    entity_idx = list(ent2idx.keys())
    relations_idx = list(rel2idx.keys())
    for i in range(len(entity_idx)):
        entity_idx[i] = int(entity_idx[i])
    for i in range(len(relations_idx)):
        relations_idx[i] = int(relations_idx[i])
    entiy_id_embeddings = np.array(entity_idx)
    relations_id_embeddings = np.array(relations_idx)

    # The index between UrbanKG id and embedding        实体ID和embedding之间的映射
    np.savetxt(path + "/relations_idx_embeddings.csv", relations_id_embeddings, encoding="utf-8", delimiter=",")
    np.savetxt(path + "/entity_idx_embedding.csv", entiy_id_embeddings, encoding="utf-8", delimiter=",")

    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0) #上下拼接
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    data_path = DATA_PATH
    dataset_name='bj_xiaorong_wo_borough'
    dataset_path = os.path.join(data_path, dataset_name)

    #处理输入
    dataset_examples, dataset_filters = process_dataset(dataset_path, dataset_name)

    #下面存储数据
    for dataset_split in ["train", "valid", "test"]:
        save_path = os.path.join(dataset_path, dataset_split + ".pickle")
        with open(save_path, "wb") as save_file:
            pickle.dump(dataset_examples[dataset_split], save_file)
    with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
        pickle.dump(dataset_filters, save_file)
