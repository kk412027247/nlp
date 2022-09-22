from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

mode_name = "bert-base-uncased"
# mode_name = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(mode_name)
model = AutoModelForTokenClassification.from_pretrained(mode_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)

