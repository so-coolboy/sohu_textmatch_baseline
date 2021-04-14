import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
flag = 'A'   # A表示对主题进行预测，B表示对事件进行预测
BERT_PATH = "hfl/chinese-bert-wwm-ext"
MODEL_PATH = str(flag)+'_'+"model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
