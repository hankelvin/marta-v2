# -*- coding: utf-8 -*-
from a2_parsers import _parse_rawtext2consttree, _parse_segmenttokenize_en
import re, os, json

########## json to CoNLL format converter ##########
# forked from CoNLL 2015 Shared Task organisers    #
# original: https://github.com/attapol/conll15st/blob/master/converter.py 


def convert_parse_json_to_conll(parse_dict):
	"""
	Convert a parse dictionary (pdtb_parses.json) into conll dictionary
	Example:
		import json
		parses = json.load('./conll15-st-03-04-15-dev/pdtb-parses.json')
		d = convert_parse_json_to_conll(parses)
	Returns: dictionary mapping from doc_id to conll format string
	"""
	from collections import OrderedDict
	import sys
	import json

	doc_id_to_conll_string = {}
	for doc_id, doc in parse_dict.items():
		token_id_offset = 0
		conll_string = ''
		for si, s in enumerate(doc['sentences']):
			print(doc)
			tokens = [t for t in s['words']]

			for i, token in enumerate(tokens):
				fields = []
				fields.append(str(i + token_id_offset))
				fields.append(str(si))
				fields.append(str(i))

				fields.append(token[0])
				fields.append(token[1]['PartOfSpeech'])
				conll_string += '\t'.join(fields)
				conll_string += '\n'
			token_id_offset += len(tokens)
			conll_string += '\n'
		doc_id_to_conll_string[doc_id] = conll_string
	
	return doc_id_to_conll_string


def convert_parse_conll_to_json(parse_dict):
	"""
	Convert a series of conll parses (wsj_####.conllu) into conll dictionary. Modification, 
	and on top, of https://github.com/attapol/conll15st/blob/master/converter.py 
	Example:
		import json
		parses = json.load('./conll15-st-03-04-15-dev/pdtb-parses.json')
		d = convert_parse_json_to_conll(parses)
	Returns: dictionary mapping from doc_id to conll format string
	"""

	pass 


########## Helper functions for handling creating Parse-class objects from raw text ##########

##### 1. Helper functions for handling conversion from PTB to UD #####

def extract_PTBparses(lang, datasets = ['train', 'dev', 'test'], 
	input_filepath = './03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/parses.json', 
	output_filepath ='./03_data/{}/stree_parses/{}_stree_PTB/{}'):
	"""
	"""

	for dataset in ['train', 'dev', 'test']:
		__filename = input_filepath.format(lang, dataset)

		with open(__filename) as f:
			__parsedict = json.load(f)
		
		for parse_num in __parsedict: 
			__sentences = __parsedict[parse_num]['sentences']
			__file_name = output_filepath.format(lang, dataset, parse_num)
			os.makedirs(os.path.dirname(__file_name), exist_ok=True)
			with open(__file_name, 'a+') as f1:
				for sent_num in range(len(__sentences)):
					f1.write("\n##&&##_{}\n".format(sent_num))
					f1.write(__sentences[sent_num]['parsetree'])
	
	return 'PTB parses extracted.'

def add_deppostags(lang, dataset, format_ ='UD1', 
	filepath_depconllu = './03_data/{}/stree_parses/{}_conllu_{}/', 
	filepath_parsedict='./03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill'):
	"""
	
	"""
	import glob, dill, os, re 

	# load the relevant parse dict 		
	with open(filepath_parsedict.format(lang, dataset), 'rb') as f1:
		Parse_dict = dill.load(f1)

    # using glob, extract the file names in the folder containing the Universal 
	# Dependencies/Stanford Dependencies postags. prior steps save the files in the folders 
	# with the 'wsj_' prefix. 
	filenames = glob.glob(filepath_depconllu.format(lang, dataset, format_)+'wsj_*')
	for filename in filenames: 
		wsj_num = os.path.basename(filename)
		with open(filepath_depconllu.format(lang, dataset, format_)+wsj_num, 'r') as f2:
			sent_id = 0
			Parse_obj = Parse_dict[wsj_num][sent_id]
			lines = f2.readlines()
		for line in lines: 
			if line !='\n':
				line_split = line.split()
				token_num = int(line_split[0])-1 # convert str data in conll to int and -1 to 
				match zero-index in our json format
				dep_postag = line_split[3]
				word_form = line_split[1]
				
				print(dep_postag)
				# the 'words' set in the CoNLL 15/16  captures the surface 
				# wordform of the sentence, i.e. not lowercased, not lemmatized or stemmed. 
				# We assert to ensure that the word is correct. we also assert that the dictionary 
				# holding the wordform's information is present. 
				
				print (wsj_num, sent_id, token_num, Parse_obj.Words[token_num][0], word_form, 
				isinstance(Parse_obj.Words[token_num][-1], dict))
				try: 
					assert Parse_obj.Words[token_num][0] == word_form and isinstance(Parse_obj.Words[token_num][1], dict) 
				except: 
					assert re.search(r'[/()\*]', word_form)
					# in the JSON format of the CoNLL 2016 dataset, the wordform stored for 
					# punctuation is PTB style and with escape char for special chars (e.g. -LRB- and \*)

				# we store the UD/SD pos tags under the 'words' set instead of the 'dependencies' set, 
				# because (i) 'words' contain a dictionary can be easily updated without breaking existing/future code. 
				Parse_obj.Words[token_num][1].update({'PartOfSpeechDepend' : dep_postag})

			else: 
				if sent_id < len(Parse_dict[wsj_num])-1: sent_id += 1
				print (sent_id, len(Parse_dict[wsj_num]))
				Parse_obj = Parse_dict[wsj_num][sent_id]
				pass 

	
	return Parse_dict

##### 2. Helper functions to obtain missing parses in CoNLL 2016 dataset and create missing Parse-class objects #####

def _make_wordsattr(document, usage = 'experiments'): 
	"""
	Given a document, containing one or more sentence, 
	
	The document is passed whole into the sentence segmenter and tokeniser (without pre-processing) 
	so as to preserve token index location. This is important for scoring purposes (to ensure that token 
	and sentence spans are always aligned). 

	"""
	
	annotated = _parse_segmenttokenize_en(document, usage=usage)

	# 1. Pre-treat: remove all the 'index'
	[annotated['sentences'][i].pop('index') for i in range(len(annotated))]

	# 2a. Specific to raw files for PTB: pop sentence if it only has 1 token and it is '.'
	# this relates to the .START \n\n head for every PTB raw file. 
	_to_pop = []
	for i in range(len(annotated['sentences'])):
		_sentence = annotated['sentences']
		if len(_sentence[i]['tokens']) == 1 and _sentence[0]['tokens'][0]['originalText'] == '.':
			_to_pop.append(i)
	[annotated['sentences'].pop(i) for i in _to_pop]
			
	# 2b. Specific to raw files for PTB: pop "START" if it has before': '', 'after': ' \n\n'
	for i in range(len(annotated['sentences'])):
		_to_pop = []
		for i2 in range(len(annotated['sentences'][i]['tokens'])):
			_token = annotated['sentences'][i]['tokens'][i2]
			if _token['word']=='START' and _token['before'] =='' and '\n\n' in _token['after']:
				_to_pop.append(i2)
		[annotated['sentences'][i]['tokens'].pop(i3) for i3 in _to_pop] 

	# 3a. add 'word' to the start of the list
	for i in range(len(annotated['sentences'])):
		for i2 in range(len(annotated['sentences'][i]['tokens'])):
			_token = annotated['sentences'][i]['tokens'][i2].copy()
			annotated['sentences'][i]['tokens'][i2] = list([_token['word'], _token])

	# 3b. pop 'word', 'originalText', 'index', 'before', and 'after'
	for i in range(len(annotated['sentences'])):
		to_pop = []
		for i2 in range(len(annotated['sentences'][i]['tokens'])):
			_token = annotated['sentences'][i]['tokens'][i2][1]

			_token['Linkers'] = list()
			_token['PartOfSpeech'] = _token['pos']
			
			_to_pop = ['index', 'word', 'originalText', 'pos', 'before', 'after']
			[_token.pop(i3) for i3 in _to_pop]
        
	return annotated
 

def _make_missingPTBparse(lang, dataset, file_num, sent_num, 
	conll_filepath='./03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/conll_format/{}.conll'):
	"""
	Helper function to obtain PTB constituency parses missing within the CoNLL 2016 json file. 
	Also used when creating Parse-class objects for (i) entire PTB wsj files missing from the 
	dataset, (ii) new sentences. 
	input | lang: str - 'en', 'fr', or 'zh'; dataset:str - 'train', 'dev', 'test' or other naming 
	conventions used for organising the split datasets; sent_num:str - the sentence offset number 
	(0-indexed); conll_filepath; str - the filepath where the conll files are stored, with placeholder 
	braces for lang, dataset, and file_num
	output | str - in bytes, the results from parsing the sentence through the Berkeley Parser 
	(Petrov and Klein, 2007)
	"""
	# using the CoNLL files provided by the Shared Task organisers to ensure sentence alignment.
	with open(conll_filepath.format(lang, dataset, file_num)) as f: 
		lines = f.readlines()
	
	while sent_num>=0:
		sentlines = list()
		curr = lines.pop(0)
		while curr != '\n': 
			sentlines.append(curr) 
			curr = lines.pop(0)
		sent_num-=1
	
	# this method will introduce a space before every single punctuation (i.e. it will introduce 
	# a slight deviation from the surface form). However, minor random sampling of the outputs 
	# from the Berkeley Parser 
	sentence = ' '.join([i.split('\t')[3] for i in sentlines])
	# sentence = __remove_whitespace(sentence) # not necessary since we are joining already-tokenised 
	# elements and 'tokenize' option is not specified in the parser within the _parse_rawtext2consttree function. 
	print(sentence)
	PTBparse = _parse_rawtext2consttree(lang, sentence, tokenized=True).rstrip(b'\n')

	return PTBparse

def __remove_whitespace(sentence):
	"""
	Helper function, used by _ma
	"""
	sentence = re.sub(r'\s([\']\w)', r'\1', sentence) 
	# handle <space>'<clitic>. important to add \ before [ and ] in punctuation list 
	sentence = re.sub(r'(\w+)\s(\w[\']\w)', r'\1\2', sentence) 
	# handle <root><space><clitic>'<clitic>. e.g. did n't
	sentence = re.sub(r'(\w+)\s([#$%&\'*+,-/:;<=>@\[\\\]^_`{|}~]+\s\w+)', r'\1\2', sentence)  
	# handle punctuation, within sentence
	sentence = re.sub(r'(\w+)\s([\'`.!?"]+)\s*', r'\1\2', sentence) 
	# handle punctuation, end of sentence, but not abbr, e.g. U.S. 

	return sentence 




##########  ##########

if __name__ == "__main__":
	
	### Run updated_parsedict functions 

	import dill 

	LANG = 'en'
	FORMAT = "SD"
	for dataset in ['train', 'dev', 'test']:

		updated_parsedict = add_deppostags('en', dataset, format_ = FORMAT, 
		filepath_depconllu = './03_data/{}/stree_parses/{}_conllu_{}/', 
		filepath_parsedict='03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill')

		if dataset == 'test': file_num = 'wsj_2301'
		if dataset == 'dev': file_num = 'wsj_2202'
		if dataset == 'train': file_num = 'wsj_0201'
		print('Processed', dataset, updated_parsedict[file_num][0].Words)
		
		with open('03_data/{}/pdtb_conll_data/ParsePDTB_dict_with{}_{}.dill'.format(LANG, dataset, FORMAT), 'wb') as f:
			dill.dump(updated_parsedict, f)
			
		with open('./03_data/{}/pdtb_conll_data/ParsePDTB_dict_with{}_{}.dill'.format(LANG, dataset, FORMAT), 'rb') as f:
			__ = dill.load(f)

		print('Loaded', dataset, __[file_num][0].Words)
	
	with open('./03_data/en/pdtb_conll_data/conll16st-en-03-29-16-train/raw/wsj_0202', 'r+', encoding='ascii') as f:
		document = f.read()
	wordsattrs = _make_wordsattr(document, usage = 'production')
	print(wordsattrs)
