# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer as BertTokenizer
from transformers import AutoModel

bert_name = 'prajjwal1/bert-tiny'
tokenizer = BertTokenizer.from_pretrained(bert_name)
model = AutoModel.from_pretrained(bert_name)

save_dir = './pretrained-model'
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
