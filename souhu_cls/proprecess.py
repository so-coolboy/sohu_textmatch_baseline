import pandas as pd
import numpy as np
import os
import json
from utils import process,write_data
from tqdm import tqdm


data_path = '/home/xiachuankun/NLP/data/sohu2021_open_data_clean/'
save_path = 'input/'
path_list = os.listdir(data_path)
path_list.remove('sample_submission.csv')
print(path_list)


train_data_a = []
val_data_a = []
test_data_a = []
train_data_b = []
val_data_b = []
test_data_b = []
for path in path_list:
    with open(data_path + path + "/train.txt", "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            data = json.loads(line.strip())
            if "A" in path:
                train_data_a.append({'source':process(data["source"]), 'target':process(data["target"]), 'label':process(data["labelA"])})
            else:
                train_data_b.append({'source':process(data["source"]), 'target':process(data["target"]), 'label':process(data["labelB"])})
    with open(data_path + path + "/valid.txt", "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            data = json.loads(line.strip())
            if "A" in path:
                val_data_a.append({'source':process(data["source"]), 'target':process(data["target"]), 'label':process(data["labelA"])})
            else:
                val_data_b.append({'source':process(data["source"]), 'target':process(data["target"]), 'label':process(data["labelB"])})
    with open(data_path + path + "/test_with_id.txt", "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            data = json.loads(line.strip())
            if "A" in path:
                test_data_a.append({'source':process(data["source"]), 'target':process(data["target"]), 'id':process(data["id"]), 'label':0})
            else:
                test_data_b.append({'source':process(data["source"]), 'target':process(data["target"]), 'id':process(data["id"]), 'label':0})




train_A = pd.DataFrame(train_data_a)
valid_A = pd.DataFrame(val_data_a)
test_A = pd.DataFrame(test_data_a)

train_B = pd.DataFrame(train_data_b)
valid_B = pd.DataFrame(val_data_b)
test_B = pd.DataFrame(test_data_b)

train_A.to_csv(save_path+'train_A.csv', index=False)
valid_A.to_csv(save_path+'valid_A.csv', index=False)
test_A.to_csv(save_path+'test_A.csv', index=False)
train_B.to_csv(save_path+'train_B.csv', index=False)
valid_B.to_csv(save_path+'valid_B.csv', index=False)
test_B.to_csv(save_path+'test_B.csv', index=False)
