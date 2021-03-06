import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import string
import numpy as np

def readfiles(filename):
	with open(filename, 'r')as f:
		c = f.readlines()
	data = [x.strip().split(' ') for x in c]
	cat_dict =  {}
	for i in data:
		cat_dict[i[0]]= i[1:]
	return cat_dict
	

def getCategoryBERT(cat_dict, tokenizer, vec_file):
	cat_vec = {}
	for k,v in cat_dict.items():
		print('-------------- inside final averaging of categories ----------------')
		print('name of the entity -->',k)
		lst_marked_text = []
		lst_tensor_token = []
		lst_segments_tensors = []
		for idx, i in enumerate(v):
			print('category no: '+ str(idx) + ' category: '+i)
			clean_i = i.replace("_"," ")
			print('cleaned category ', clean_i)
			marked_text = "[CLS] " + clean_i + " [SEP]"
			print('marked text', marked_text)
			lst_marked_text.append(marked_text)
			tokenized_text = tokenizer.tokenize(marked_text)
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
			segments_ids = [idx+1] * len(tokenized_text)
			tokens_tensor = torch.tensor([indexed_tokens])
			lst_tensor_token.append(tokens_tensor)
			segments_tensors = torch.tensor([segments_ids])
			lst_segments_tensors.append(segments_tensors)
		print('length of segment tensors',len(lst_segments_tensors))
		res = zip(lst_tensor_token, lst_segments_tensors)
		# Load pre-trained model (weights)
		model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True,
		)
		model.eval()
		lst_hidden_states =[]
		print('length of res', res)
		with torch.no_grad():
			for i in res:
				outputs = model(i[0], i[1])
				hidden_states = outputs[2]
				lst_hidden_states.append(hidden_states)
		print(len(lst_hidden_states))
		print('Type of hidden_states: ', type(hidden_states))
		lst_token_embeddings = []
		for i in lst_hidden_states:
			token_embeddings = torch.stack(i, dim=0)
			token_embeddings.size()
			lst_token_embeddings.append(token_embeddings)
		print('Number of token embeddings: ', len(lst_token_embeddings))
		token_embed_lst =[]
		for i in lst_token_embeddings:
			token_embeddings = torch.squeeze(i, dim=1)
			print(token_embeddings.size())
			token_embed_lst.append(token_embeddings)

		
		lst_token_embeddings = []
		for i in token_embed_lst:
			token_embeddings = i.permute(1,0,2)
			print(token_embeddings.size())
			lst_token_embeddings.append(token_embeddings)
		for i in lst_token_embeddings:
			print(i.size())

		
		lst_token_vecs_cat =[]
		for i in lst_token_embeddings:
			token_vecs_cat = []
			for token in i:
				cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
				token_vecs_cat.append(cat_vec)
			lst_token_vecs_cat.append(token_vecs_cat)

		print('total number of words/phrases: ', len(lst_token_vecs_cat))

		print ('Shape is: %d x %d' % (len(lst_token_vecs_cat[0]), len(lst_token_vecs_cat[0][0])))
		lst_token_vecs_sum = []
		for i in lst_token_embeddings:
			token_vecs_sum = []
			for token in i:
				sum_vec = torch.sum(token[-4:], dim=0)
				token_vecs_sum.append(sum_vec)
			lst_token_vecs_sum.append(token_vecs_sum)
		print('Number of vectors: ', len(lst_token_vecs_sum))
		print ('Shape is: %d x %d' % (len(lst_token_vecs_sum[0]), len(lst_token_vecs_sum[0][0])))
		lst_sentence_embedding = []
		for i in lst_hidden_states:
			token_vecs = i[-2][0]
			sentence_embedding = torch.mean(token_vecs, dim=0)
			lst_sentence_embedding.append(sentence_embedding.numpy())
		print('no of sentence embeddings ',len(lst_sentence_embedding))
		print('dimension of each sentence vector',len(lst_sentence_embedding[0]))
		category_vec = np.mean(lst_sentence_embedding,  axis=0)
		print('length of category vector',len(category_vec))
		vec_file.write(k+' '+' '.join([str(elem) for elem in category_vec]))
		vec_file.write('\n')
	return 0

if __name__ == "__main__":
	
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	
	train_cat_dict = readfiles('Etrain_cat')
	train_vec = open('BERT_train','w')
	getCategoryBERT(train_cat_dict,tokenizer,train_vec)
	
	test_cat_dict = readfiles('Etest_cat')
	test_vec = open('BERT_test','w')
	getCategoryBERT(test_cat_dict,tokenizer,test_vec)
	
	dev_cat_dict = readfiles('Edev_cat')
	dev_vec = open('BERT_dev','w')
	getCategoryBERT(dev_cat_dict,tokenizer,dev_vec)


