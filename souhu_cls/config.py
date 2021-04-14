import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
flag = 'B'   # A表示对主题进行预测，B表示对事件进行预测
BERT_PATH = "hfl/chinese-bert-wwm-ext"
MODEL_PATH = str(flag)+'_'+"model.bin"
TRAINING_FILE = "input/train_cls.csv"
TEST_FILE = "input/test_cls.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)