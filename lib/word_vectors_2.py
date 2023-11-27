"""
Adapted from PyTorch's text library.
"""

import array
import os
import zipfile

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
import numpy as np
import nltk

nltk.download('averaged_perceptron_tagger')

#from nltk import pos_tag

def find_special_character(word_list):
	special_characters = {"underscore": "_", "space": " ", "slash": "/"}
	
	for word in word_list:
		for key, character in special_characters.items():
			if character in word:
				return key
	return None

def is_verb(word):
	# Verbs are tagged with 'VB' or 'VBP' in the Penn Treebank tagset
	return word[1].startswith('VB')

def obj_edge_vectors(names, wv_type='glove.6B', wv_dir=None, wv_dim=300):
	wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

	vectors = torch.Tensor(len(names), wv_dim)

	vectors.normal_(0,1)

	
	for i, token in enumerate(names):
		
		lw_token=[]
		if token=="__background__":
			token="background"

		else:	

			character=find_special_character(token)
			if character is "underscore":
				list_token=sorted(token.split('_'), key=lambda x: len(x), reverse=True)
				for k in list_token:
				 	lw_token.append(k)
				#lw_token.append( sorted(token.split('_'), key=lambda x: len(x), reverse=True)[1])
			elif character is "space":	

				lw_token.append( sorted(token.split(' '), key=lambda x: len(x), reverse=True)[1])

			elif character is "slash":	

				lw_token.append(token.split('/')[0])
			else:
				#lw_tok.append(token)
				lw_token.append(token)

			#print("lw_token",lw_token)
			lw_token1=[]
			wv_index1=[]
			#print("lw_token",lw_token)
			l=len(lw_token)
			#print("len",l)
			#print("length",l)

			'''
			if len(lw_token)>1:
				#print("lw_token",lw_token)
				pos_tags = pos_tag(lw_token)
				#print("pos_tags",pos_tags)

			
				for j in range(len(pos_tags)):	
					# if is_verb(pos_tags[j]):
					# 	continue
					# else:
						lw_token1.append(pos_tags[j][0])'''	
			#else:
			lw_token1=lw_token	
				#print("lw_token1",lw_token1)		




			for j in lw_token1:
				

				wv_index1.append(wv_dict.get(j))

			
				

			if wv_index1[0] is not None:
				print(f"{token}--> {lw_token1}")
				vec=[]
				for k in wv_index1:
					#print(wv_arr[wv_index1[k]].shape)
					vec.append( wv_arr[k].cpu().numpy())
						
						
				vec2=np.array(vec)
				#print(vec2.shape)

				vec3=np.mean(vec2,axis=0)
				#print(vec3.shape)
				vec4=torch.tensor(vec3)
				
				vectors[i]=vec4
				##print("background")

			else:
				print("failure")	
		'''	
		wv_index = wv_dict.get(token.split('/')[0], None)
		wv_index1=[]
		vec=[]
		if wv_index is not None:
			vectors[i] = wv_arr[wv_index]
			print(f"embedding of {token} done")

		########### edited by Rinki    
		else:
			lw_token1 = sorted(token.split('_'), key=lambda x: len(x), reverse=True)

			for j in range(len(lw_token1)):
				#print(lw_token1[j])
				wv_index1.append(wv_dict.get(lw_token1[j]))
				

			if wv_index1[0] is not None:
				print(f"all words embeddings of {token} are averaged")
				for k in range(len(wv_index1)):
					#print(wv_arr[wv_index1[k]].shape)
					vec.append( wv_arr[wv_index1[k]].cpu().numpy())
						
						
				vec2=np.array(vec)
				#print(vec2.shape)

				vec3=np.mean(vec2,axis=0)
				#print(vec3.shape)
				vec4=torch.tensor(vec3)
				
				vectors[i]=vec4

			else:    

				 
				#Try the longest word (hopefully won't be a preposition)
				lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
				print("{} -> {} ".format(token, lw_token))
				wv_index = wv_dict.get(lw_token, None)
				if wv_index is not None:
					vectors[i] = wv_arr[wv_index]
				else: 	
					print("fail on {}".format(token))'''
	
	return vectors

URL = {
		'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
		'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
		'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
		'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
		}


def load_word_vectors(root, wv_type, dim):
	"""Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
	if isinstance(dim, int):
		dim = str(dim) + 'd'
	fname = os.path.join(root, wv_type + '.' + dim)
	if os.path.isfile(fname + '.pt'):
		fname_pt = fname + '.pt'
		print('loading word vectors from', fname_pt)
		try:
			return torch.load(fname_pt)
		except Exception as e:
			print("""
				Error loading the model from {}

				This could be because this code was previously run with one
				PyTorch version to generate cached data and is now being
				run with another version.
				You can try to delete the cached files on disk (this file
				  and others) and re-running the code

				Error message:
				---------
				{}
				""".format(fname_pt, str(e)))
			sys.exit(-1)
	if os.path.isfile(fname + '.txt'):
		fname_txt = fname + '.txt'
		cm = open(fname_txt, 'rb')
		cm = [line for line in cm]
	elif os.path.basename(wv_type) in URL:
		url = URL[wv_type]
		print('downloading word vectors from {}'.format(url))
		filename = os.path.basename(fname)
		if not os.path.exists(root):
			os.makedirs(root)
		with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
			fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
			with zipfile.ZipFile(fname, "r") as zf:
				print('extracting word vectors into {}'.format(root))
				zf.extractall(root)
		if not os.path.isfile(fname + '.txt'):
			raise RuntimeError('no word vectors of requested dimension found')
		return load_word_vectors(root, wv_type, dim)
	else:
		raise RuntimeError('unable to load word vectors')

	wv_tokens, wv_arr, wv_size = [], array.array('d'), None
	if cm is not None:
		for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
			entries = cm[line].strip().split(b' ')
			word, entries = entries[0], entries[1:]
			if wv_size is None:
				wv_size = len(entries)
			try:
				if isinstance(word, six.binary_type):
					word = word.decode('utf-8')
			except:
				print('non-UTF8 token', repr(word), 'ignored')
				continue
			wv_arr.extend(float(x) for x in entries)
			wv_tokens.append(word)

	wv_dict = {word: i for i, word in enumerate(wv_tokens)}
	wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
	ret = (wv_dict, wv_arr, wv_size)
	torch.save(ret, fname + '.pt')
	return ret

def reporthook(t):
	"""https://github.com/tqdm/tqdm"""
	last_b = [0]

	def inner(b=1, bsize=1, tsize=None):
		"""
		b: int, optionala
		Number of blocks just transferred [default: ĺeftright].
		bsize: int, optional
		Size of each block (in tqdm units) [default: ĺeftright].
		tsize: int, optional
		Total size (in tqdm units). If [default: None] remains unchanged.
		"""
		if tsize is not None:
			t.total = tsize
		t.update((b - last_b[0]) * bsize)
		last_b[0] = b
	return inner
