import config
import dataset
import engine
import torch
import pandas as pd
from model import BERTBaseUncased
import time

def get_time():
    return time.strftime("%Y%m%d_%H%M",time.localtime())

# 先预测A类
config.MODEL_PATH = 'A'+'_'+"model.bin"
df_test1 = pd.read_csv(f'input/test_A.csv')


test_dataset = dataset.BERTDataset(df_test1)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4, shuffle=False
)

device = torch.device(config.DEVICE)
model = BERTBaseUncased(num_class=2)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)


_, fin_outputs = engine.eval_fn(test_data_loader, model, device)
df_test1['label'] = fin_outputs

print(df_test1.head())
print(df_test1.label.value_counts())


# 再预测B类
config.MODEL_PATH = 'B'+'_'+"model.bin"
df_test2 = pd.read_csv(f'input/test_B.csv')


test_dataset = dataset.BERTDataset(df_test2)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4, shuffle=False
)

device = torch.device(config.DEVICE)
model = BERTBaseUncased(num_class=2)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)


_, fin_outputs = engine.eval_fn(test_data_loader, model, device)
df_test2['label'] = fin_outputs

print(df_test2.head())
print(df_test2.label.value_counts())


result1 = df_test1[['id', 'label']]
result2 = df_test2[['id', 'label']]

result = result1.append(result2)

result.to_csv(f'result/result_{get_time()}.csv', index=False)


