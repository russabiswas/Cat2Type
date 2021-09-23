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
        #coversion = glove2word2vec(embedding_file, wiki2vecfile)
        model = KeyedVectors.load_word2vec_format(embedding_file)
        return model


def avgvectors(category,model):
        for c in string.punctuation:
                category = category.replace(c," ").lower()
                category = " ".join(category.split())
        tmp = []
        print('category name--------')
        print(category)
        for i in category.split(' '):
                try:
                        tmp.append(model[i])
                except:
                        print('inside except')
                        z= np.zeros((300,), dtype=int)
                        tmp.append(z)
        print('no of vectors for the word, ',category, 'is, ', len(tmp))
        vec_avg = np.mean(tmp, axis=0)
        return vec_avg

def getVectors(cat_dict, wiki2vec_embed, op_vectorsFile):
	cat_vec = {}
	for k,v in cat_dict.items():
		tmp = []
		print('Russa -->',k)
		for i in v:
			word = avgvectors(i,wiki2vec_embed)
			tmp.append(word)
		cat_avg = np.mean(tmp, axis=0)
		op_vectorsFile.write(k+ '\t' +str(len(tmp)) +'\t'+ ' '.join([str(elem) for elem in word]))
		op_vectorsFile.write('\n')
		cat_vec[k] = cat_avg
	return cat_vec




if __name__ == "__main__":
	embed = loadmodel('wiki2vecVecfile.txt')
	train_cat_dict = readfiles('Etrain_cat')
	train_vectorsFile_avg = open('wiki2vec_AVG_train_catVectors','w')
	train_cat_vec = getVectors(train_cat_dict, embed, train_vectorsFile_avg)

	test_cat_dict = readfiles('Etest_cat')
	test_vectorsFile_avg = open('wiki2vec_AVG_test_catVectors','w')
	test_cat_vec = getVectors(test_cat_dict, embed, test_vectorsFile_avg)

	dev_cat_dict = readfiles('Edev_cat')
	dev_vectorsFile_avg = open('wiki2vec_AVG_dev_catVectors','w')
	dev_cat_vec = getVectors(dev_cat_dict, embed, dev_vectorsFile_avg)
