import collections
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config import Config
from models import UL_Manualdown

def train_test_datasets(train_project_paths, test_project_path):
    # sampled_train_project_paths = train_sampling(train_project_paths) if not Config.ALL_DATA else train_project_paths
    train_dfs, train_targets, train_data, train_tloc = create_datasets(train_project_paths)
    test_dfs, test_targets, test_data, test_tloc = create_datasets([test_project_path])
    return train_dfs, train_targets, train_data, train_tloc, \
            test_dfs, test_targets, test_data, test_tloc

def create_datasets(project_paths, random_seed=0):
    dfs = []
    targets = []
    datasets = []
    random.seed(random_seed)

    for project_path in project_paths:
        df = pd.read_csv(project_path).sample(frac = 1, random_state=random_seed)
        df = df[df[datasets_describe[group]['loc_colname']] != 0]
        tloc = df[datasets_describe[group]['loc_colname']]
        df = df.drop(columns=datasets_describe[group]['string_colnames'])
        df = (df-df.min())/(df.max()-df.min())
        df = df.dropna()
        dfs.append(df)
        target = df.pop(datasets_describe[group]['target_colname'])
        targets.append(target)
        datasets.append(tf.data.Dataset.from_tensor_slices(({'features':df.values,'target':np.array(target.values)})))
    return dfs, targets, datasets, tloc

if __name__=="__main__":
    group_names = ['AEEEM', 'METRICSREPO', 'RELINK', 'SOFTLAB']
    for group in group_names:
        with open(Config.DATA_PATH + 'datasets_describe.json', 'r') as f:
            datasets_describe = json.load(f)

        all_project_paths = glob.glob(Config.DATA_PATH + group + '/*.csv')
        test_project_names = datasets_describe[group]['latest_versions']

        for i in range(len(datasets_describe[group]['latest_versions'])):
            test_project_name = test_project_names[i]
            print(f'\ntest project: {os.path.basename(test_project_name)}\n')
            base_test = re.sub('[\d+].[\d+].csv', '', test_project_name)
            test_project_path = os.path.join(Config.DATA_PATH + group + '/' + test_project_name)

            print('UL_Manualdown')
            out_dict = collections.OrderedDict()
            model_result = UL_Manualdown(*create_datasets([test_project_path]), test_project_name)
            out_dict.update(model_result)
            result = pd.DataFrame(out_dict, index=out_dict[list(out_dict.keys())[0]].keys()).transpose()
            print(result)