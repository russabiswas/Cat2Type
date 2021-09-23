import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
import string


def readfiles(filename):
	with open(filename, 'r')as f:
		c = f.readlines()
	data = [x.strip().split(' ') for x in c]
	cat_dict =  {}
	for i in data:
		cat_dict[i[0]]= i[1:]
	return cat_dict


def readfilePandas(filename):
        df = pd.read_csv(filename, engine='python', header=None, names=['entity'])
        t = df.values.tolist()
        text = [item for sublist in t for item in sublist]
        return text

def loadmodel(embedding_file):
        model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        return model

def avgvectors(category,model):
	for c in string.punctuation:
		category = category.replace(c," ").lower()
		category = " ".join(category.split())
	tmp = []
	print('------------inside average of each of the categories--------------')
	print('category name--------')
	print(category)
	for i in category.split(' '):
		print(i)
		try:
			tmp.append(model[i])
			print(i +' exists')
		except:
			print(i + ' does not exist')
			print('inside except')
			z= np.zeros((300,), dtype=int)
			tmp.append(z)
	print('no of vectors for the word, ',category, 'is, ', len(tmp))
	vec_avg = np.mean(tmp, axis=0)
	return vec_avg

def getVectors(cat_dict, embed, op_vectorsFile):
	cat_vec = {}
	for k,v in cat_dict.items():
		tmp = []
		print('-------------- inside final averaging of categories ----------------')
		print('name of the entity -->',k)
		for idx, i in enumerate(v):
			print('category no: '+ i+ ' category: '+i)
			word = avgvectors(i, embed)
			tmp.append(word)
		print('total number of categories ',len(tmp))
		print('total number of categories ', len(v))
		cat_avg = np.mean(tmp, axis=0)
		op_vectorsFile.write(k+ '\t' +str(len(tmp)) +'\t'+ ' '.join([str(elem) for elem in word]))
		op_vectorsFile.write('\n')
		cat_vec[k] = cat_avg
	return cat_vec




if __name__ == "__main__":
	embed = loadmodel('../GoogleNews-vectors-negative300.bin')
	train_cat_dict = readfiles('Etrain_cat')
	train_vectorsFile_avg = open('w2v_train_catVectors','w')
	train_cat_vec = getVectors(train_cat_dict, embed, train_vectorsFile_avg)

	test_cat_dict = readfiles('Etest_cat')
	test_vectorsFile_avg = open('w2v_test_catVectors','w')
	test_cat_vec = getVectors(test_cat_dict, embed, test_vectorsFile_avg)
	dev_cat_dict = readfiles('Edev_cat')
	dev_vectorsFile_avg = open('w2v_dev_catVectors','w')
	dev_cat_vec = getVectors(dev_cat_dict, embed, dev_vectorsFile_avg)
