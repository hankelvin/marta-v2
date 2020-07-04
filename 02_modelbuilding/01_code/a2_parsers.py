# -*- coding: utf-8 -*-
import subprocess, re
from os import chdir, getcwd, environ
from stanfordnlp.server import CoreNLPClient

"""
Helper functions for handling creating Parse-class objects from raw text. 
"""
	
def _parse_segmenttokenize_en(document, usage = 'experiments'): 
	"""
	Given a document, in str format, containing one or more sentences, returns a set of segmented 
	and tokenized strings, with indexing information. This format is the basis for: (i) the format 
	for storing information on sentences and tokens in the CoNLL 2015 and 2016 Shared Task; 
	(ii) the .Words attribute in Parse-class objects. This function uses the stanford-corenlp 
	package and requires the CoreNLP Java package to be downloaded and built (with Ant or Maven) 
	and saved to the 04_utils folder. 
	"""
	cwd =  getcwd()
	if usage == 'production' or 'experiments': 
		version = 'stanford-corenlp-full-2018-10-05/'

		corenlp_path = re.findall(r'\S*/marta-v2', cwd)[0] + '/04_utils/' + version
		environ["CORENLP_HOME"] = corenlp_path

		with CoreNLPClient(annotators="tokenize ssplit pos".split(), memory='1G',
		be_quiet=True, max_char_length=100000,) as client:
			annotated = client.annotate(document, output_format='json')
		client.stop()

	elif usage == 'experiments_352': 
		version = 'stanford-corenlp-full-2018-10-05/' #-2015-04-20/'
		corenlp_path = re.findall(r'\S*/marta-v2', cwd)[0] + '/04_utils/' + version
		chdir(corenlp_path)
		args =  ["*", '-Xmx500m', 'edu.stanford.nlp.pipeline.StanfordCoreNLP','-annotators', 
			'tokenize,ssplit,pos', '-tokenize.whitespace', '-ssplit.eolonly', '-outputFormat', 'json', 
			'-maxLength', '10000']  # necessary to set -maxLength (default is only 200); neccessary to 
									# specify -tokenize.whitespace, since our sentence is joined from already-tokenized.
		process = subprocess.Popen(['java', '-cp']+args, stdout=subprocess.PIPE, stdin= subprocess.PIPE, 
				stderr=subprocess.STDOUT)
	
		annotated, error = process.communicate(input=document.encode('utf-8'))	
		chdir(cwd) # setting the current working directory back to original, else it causes errors downstream 
		
		# convert byte to utf-8
		annotated = annotated.decode('utf-8')

		# extract the parse sections

	return annotated


def _parse_rawtext2consttree(lang, sentence, berkparser_jarpath = '/04_utils/berkeleyparser-master/', tokenized=True ):
	"""
	Function to get constituency parses from raw text. Calls the Berkeley Parser (Petrov and Klein, 2007) 
	via command line commands, to parse a single sentences in string format. 
	The parser https://github.com/slavpetrov/berkeleyparser is configured here to return the parsed result in STDOUT.  

	"""
	import time 
	cwd = getcwd()
	
	if lang == 'en': berkparser_grammer = 'eng_sm6.gr'
	if lang == 'fr': 
		answer = input('This function utilises a model for French provided with the Berkeley Parser \
			(Petrov and Klein 2007). Enter \'Y\' if you would like to proceed, or \'N\' to abort.')
		if answer.lower() == 'y': berkparser_grammer = 'fra_sm5.gr'
		else: return 'Parsing aborted.'  
	if lang == 'zh': 
		answer = input('Note that the CoNLL 2016 Shared Task utilised a different parser to produce \
			the constituency parses in the task. Enter \'Y\' if you would like to proceed, or \'N\' to abort.')
		if answer.lower() == 'y': berkparser_grammer = 'chn_sm5.gr'
		else: return 'Parsing aborted.'   
	
	chdir(cwd + berkparser_jarpath)
	args =  ['BerkeleyParser-1.7.jar', '-accurate', '-maxLength', '750', '-gr', berkparser_grammer] 
	# necessary to set -maxLength (default is 200, but some PTB sentences are up to 600 characters long); 
	# no need to specify '-tokenize', since our sentence is joined from already-tokenized.
	process = subprocess.Popen(['java', '-jar']+args, stdout=subprocess.PIPE, 
			stdin= subprocess.PIPE, stderr=subprocess.STDOUT)

	if tokenized == False: 
		args.append('-tokenize') # 'Tokenize input first. (Default: True=text is already tokenized)'
	if lang == 'zh':
		args.append('-chinese') # 'Enable some Chinese specific features in the lexicon.'

	output, error = process.communicate(input=sentence.encode('utf-8'))	
	chdir(cwd) # setting the current working directory back to original, else it causes errors downstream 

	# convert stdout byte output to utf-8
	output = output.decode('utf-8')

	return output


def convert_const2dep(lang, dataset, filename, readpath = '/03_data/{}/stree_parses/{}_stree_PTB/', 
	writepath = '/03_data/{}/stree_parses/{}_conllu_{}/', format_ = 'UD1', usage='experiments'): 
	"""
	https://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/ud/UniversalDependenciesConverter.html

	edu.stanford.nlp.trees.ud.UniversalDependenciesConverter only takes inputs in the form of file(s) 
	containing (i) s-tree parses, or (ii) conll format. 
	"""
	if lang not in ['en', 'zh']: # french and chinese tree banks use different tagsets from PTB, therefore 
								 # UniversalDependenciesConverter may not work. 
		raise ValueError

	cwd =  getcwd()
	if usage == 'production' or usage =='experiments': version = 'stanford-corenlp-full-2018-10-05'

	relpath_to_folder = re.findall(r'\S*/marta-v2', cwd)[0]
	corenlp_path = relpath_to_folder + '/04_utils/' + version
	chdir(corenlp_path)
	readpath = relpath_to_folder + readpath.format(lang, dataset) + filename
	writepath = relpath_to_folder + writepath.format(lang, dataset, format_) + filename

	if format_ == 'UD1': format_option = ''
	if format_ == 'SD': format_option = '_sd' 
	if lang == 'en': 
		bashCommand = 'java -mx1g -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -language {}{} -conllx -treeFile {} > {}'.format(lang, format_option, readpath, writepath) 
	if lang == 'zh': 
		bashCommand = 'java -mx1g -cp "*" edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -language {}{} -treeFile {} > {}'.format(lang, format_option, readpath, writepath) 
		# see https://nlp.stanford.edu/software/stanford-dependencies.shtml and -help for 
		# 'java -mx1g -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure' and  
		# 'java -mx1g -cp "*" edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure' 
	
	process = subprocess.run(bashCommand, stdout=subprocess.PIPE, shell=True)

	chdir(cwd) # setting the current working directory back to original, else it causes errors downstream 

	return '{} parsed.'.format(filename) 

if __name__ == "__main__":
	# script tests below
		
	# print(convert_const2dep(lang = 'en', dataset = 'dev', filename = 'wsj_2200', format_ = 'SD', usage = 'experiments'))
	# with open('./03_data/en/pdtb_conll_data/conll16st-en-03-29-16-test/raw/wsj_2311', 'r') as f:
	# 	document = f.read()
	# print(document.split('\n')[2])
	# # print(_parse_rawtext2deptree(document, usage = 'experiments'))


	# lang = 'en'
	# cwd =  getcwd()
	# version = 'stanford-corenlp-full-2018-10-05'
	# corenlp_path = re.findall(r'\S*/marta-v2', cwd)[0] + '/04_utils/' + version
	# environ["CORENLP_HOME"] = corenlp_path
	# if lang == 'en': lang = 'english'
	# if lang == 'fr': lang = 'french'
	# if lang == 'zh': lang = 'chinese'
	
	
	# corenlpclient_UD1 = CoreNLPClient(properties = '', annotators="tokenize ssplit pos depparse".split(), memory='2G', be_quiet=False, max_char_length=100000,)

	# anno_UD1 = corenlpclient_UD1.annotate(document.split('\n')[2],output_format='json')

	# corenlpclient_UD1.stop()

	# print( anno_UD1['sentences'])	

	import os, dill
	import a_preprocessor
	LANG = 'en'
	dataset='train'
	with open("03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill".format(LANG,dataset), "rb") as f:
		conns = dill.load(f)

	print(conns['wsj_1057'][143].RawText)

	lang = 'en'
	# text1 = '`` What you \'re really asking is , Are the profit and loss margins anticipated on the events acceptable to management ? \'\' he says .'
	text1 = conns['wsj_1057'][143].RawText
	print(text1)
	cwd =  os.getcwd()
	version = 'stanford-corenlp-full-2018-10-05'
	corenlp_path = re.findall(r'\S*/marta-v2', cwd)[0] + '/04_utils/' + version
	os.environ["CORENLP_HOME"] = corenlp_path
	
	corenlpclient_UD1 = CoreNLPClient(properties = {'ssplit.isOneSentence': True}, annotators = ['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'udfeats'], memory='2G', be_quiet=False, max_char_length=100000, output_format = 'conllu')
	_UD1_Auto = corenlpclient_UD1.annotate(text1)
	# annotators = ['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'udfeats']
	# _UD1_Auto = _UD1_Auto['sentences'][1]['basicDependencies'] # extract only basic dependencies
	print(_UD1_Auto)
	corenlpclient_UD1.stop()

	print(convert_const2dep(LANG, dataset, filename = '', readpath = '/02_modelbuilding/02_output/input_temp.parser', writepath ='/02_modelbuilding/02_output/output_temp.parser',format_ = 'UD1', usage='experiments'))

	annotated = _parse_segmenttokenize_en('What is the star of the moon? Where is the sea of the trees?', usage='production')
	print('Annotated', annotated)
	

### To Do: 
# 1. finalise usage=experiments portion of _parse_segmenttokenize_en. (extract relevant CoreNLP 
# output from the entire stdout bytestring)
# 2. resolve sentence segmentation in experiments_352, which uses CoreNLP v3.5.2. 
# Unlike CoreNLP 3.9.2, it is now resetting the CharacterOffsetBegin/End count for new sentences in a doc. 
# Our current solution is to use 3.9.2 for both production and experiments. 