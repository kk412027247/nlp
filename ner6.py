from transformers import pipeline
from transformers import (
    TF2_WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    TFBertForTokenClassification,
    create_optimizer)
import pandas as pd

MODEL_CLASSES = {"bert": (BertConfig, TFBertForTokenClassification, BertTokenizer)}
BERT_MODEL = "bert-base-multilingual-cased"

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
df_data = pd.read_csv("data/ner_dataset.csv", sep=",", encoding="latin1").fillna(method='ffill')
tag_list = df_data.Tag.unique()
num_labels = len(tag_list) + 1
config = config_class.from_pretrained(BERT_MODEL, num_labels=num_labels)

model = model_class.from_pretrained(
    BERT_MODEL,
    from_pt=bool(".bin" in BERT_MODEL),
    config=config)

tokenizer = tokenizer_class.from_pretrained(BERT_MODEL, do_lower_case=False)

nlp = pipeline('ner', model=model, tokenizer=tokenizer)
s = "I am Joe and live in London"
print(nlp(s))
