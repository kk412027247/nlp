from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
print(output)


tokenizer2 = BertTokenizer.from_pretrained('bert-base-chinese')
model2 = TFBertModel.from_pretrained("bert-base-chinese")
text2 = "巴黎是[MASK]国的首都。"
encoded_input2 = tokenizer2(text2, return_tensors='tf')
output2 = model2(encoded_input2)
print(output2)

