import config
import torch
import pandas as pd

class BERTDataset:
    def __init__(self, data):
        self.data = data
        self.source = self.data['source']
        self.target = self.data['target']
        self.label = self.data['label']
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        source = str(self.source[item])
        target = str(self.target[item])
        

        inputs = self.tokenizer.encode_plus(
            source,
            target,
            add_special_tokens=True,
            truncation = 'longest_first',
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        token_type_ids = inputs["token_type_ids"][0]

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "targets": torch.tensor(self.label[item], dtype=torch.float),
        }



if __name__ == '__main__':
    train_A = pd.read_csv('input/valid_A.csv')
    train_A.head()
    train_set = BERTDataset(train_A)
    print(train_set[4])

    import transformers
    TOKENIZER = transformers.BertTokenizer.from_pretrained(config.BERT_PATH, do_lower_case=True)