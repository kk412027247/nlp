# https://huggingface.co/dslim/bert-base-NER

# import os
# proxy = 'http://127.0.0.1:8001'
# os.environ['http_proxy'] = proxy 
# os.environ['HTTP_PROXY'] = proxy
# os.environ['https_proxy'] = proxy
# os.environ['HTTPS_PROXY'] = proxy

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)