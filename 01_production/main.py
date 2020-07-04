import glob, sys, os, json, copy, pickle
from inspect import getmembers, isfunction
sys.path.insert(1, '02_modelbuilding/01_code/')
# import a_preprocessor 
from a1_dataloader import ParsePDTB, RelationPDTB
from a_preprocessor import get_connectivecandidates, __make_sorted_connslex
import b_featuresbuilder
from b_featuresbuilder import do_onehotencoding, make_featureset
import spacy, en_core_web_sm
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import pandas as pd

##### Notes on usage #####
# 1. This script is intended for production usage. To replicate the experiments in Pitler & Nenkova 2009, Lin et al 2014, Li et al 2016 and Han et al 2020, use the scripts in 02_modelbuilding.
# 2. This script may be used as an import, or command line (--help for more details on each argument.) 
# 3. The path to a directory containing plaintext documents to be processed should be specified. 
# 4. Results are separated into predicted connectives and non-connectives. They are saved in csv format with the intention for easy loading into a pandas dataframe.


def process_predict_connective(input_dir, fw, pdtb_v, lang, connslex, mapdict):
    """
    Given a directory where a collection of plaintext files (each a document), loads, segment, tokenise and parses each, identifies connective candidates, and produces the necessary representations (in the form of ParsePDTB and RelationPDTB instances). These representations are used to generate the featuresets in order to apply the trained models for predicting whether the candidates are (or are not) connectives.
    Parameters: 
    input_dir (str): The directory path containing the files holding the documents to be processed.
    fw (str): The syntactic framework to produce the features on, as well as apply the models on.
    pdtb_v (int): The version of PDTB to label the Parse-class instances with.
    lang (str): The language that the document(s) is(are) in.
    connslex (str):The filepath to a plaintext file containing a lexicon of connectives, each on a new line.
    mapdict (str):The filepath to a json file containing a lexicon of connectives. Each key is a possible lexicalisaton of the connectives in the lexicon, and its value is the canonical form.
    Return:
    true_list, false_list (each a list): of lists of predicted connectives/non-connectives. Information on each connective/non-connective is returned, including which document (DocID), sentence (SentID), its form (RawText), start and end character offset within the document it is found (CharStartInDoc and CharStartEndDoc). 
    """
    
    ### 1. Producing the collection of ParsePDTB instances. These are needed to run the scripts to  (1) search for connective candidates and create the RelationPDTB instances for those found; (2) generate the featuresets (with the support of the RelationPDTB instances).
    Parse_dict = make_parsedict(input_dir = input_dir, fw = fw, pdtb_v = pdtb_v, lang = lang)
    
    ### 2. Loading (1) the connectives lexicon, and (2) the mapping of different lexicalisations of a canonical form of a connective.
    with open(connslex) as f: 
        connslex_ = [i.strip('\n') for i in f.readlines()] 
    sorted_connslex = __make_sorted_connslex(connslex_) 
    with open(mapdict) as f: mapping_dict = json.load(f)
    Parse_dict_c = copy.deepcopy(Parse_dict)
    
    ### 3. Identify connective candidates and create a RelationPDTB instance for each of these
    Relations_list = get_connectivecandidates(Parse_dict_c, sorted_connslex = sorted_connslex, mapping_dict = mapping_dict, lowercase = True, posexp = False)

    ### 4. Separate treatment for PTB or UD input. (1) load the appropriate featureset building script, (2) load the appropriate pre-trained sklearn classifier and feature names. 
    if fw == 'UD2': 
        import b2_featurebuilder_udrun2 as FEATUREBUILDER
        FEATURE_FUNCS = getmembers(FEATUREBUILDER.Li_etal16_UD1, isfunction)
        with open('02_modelbuilding/01_code/models/model_UD1_Auto_Run2.model', 'rb') as f:
            model_featnames, classifier = pickle.load(f) 
    elif fw == 'PTB': 
        import b1_featurebuilder_ptbrun1 as FEATUREBUILDER
        FEATURE_FUNCS = getmembers(FEATUREBUILDER.Li_etal16, isfunction)
        with open('02_modelbuilding/01_code/models/model_PTB_Auto_Run1.model', 'rb') as f:
            model_featnames, classifier = pickle.load(f) 
    
    print('Number of candidates found', len(Relations_list))
    ### 5. Create the necessary featuresets, which are then one-hot encoded.
    df_OHE, df_labelsdict = do_onehotencoding(make_featureset(Relations_list, Parse_dict, gold = False, 
    feature_funcs = FEATURE_FUNCS, featurebuilder = FEATUREBUILDER, framework = fw))
    
    ### 6. Ensure alignment of generated features to model features.
    numfeats_model, numsamples = len(model_featnames), df_OHE.shape[0]
    df_modelfeats = pd.DataFrame(np.zeros([numsamples, numfeats_model], dtype=int), columns=model_featnames.values())
    feat_intersect = set(df_modelfeats.columns).intersection(set(df_OHE.columns))
    # set the feature values from df_OHE onto the empty df_modelfeats
    for featname in feat_intersect: df_modelfeats.loc[:, featname] = df_OHE.loc[:, featname]

    ### 7. Predict labels
    predictions = classifier.predict(df_modelfeats)
    print('Predictions made. Proceeding to save results.')

    ### 8. update RelationPDTB instance
    true_list, false_list = [], []
    for idx, RelationPDTB in enumerate(Relations_list):
        connective_info = (RelationPDTB.DocID, RelationPDTB.SentID, RelationPDTB.Connective['RawText'], 
                            RelationPDTB.Connective['CharSpanList'][0][0], RelationPDTB.Connective['CharSpanList'][0][1])
        if predictions[idx] == 1:
            RelationPDTB.PosExp = True 
            true_list.append(connective_info)
        else:
            RelationPDTB.PosExp = False
            false_list.append(connective_info)

    ### 9. Save ParsePDTB and RelationPDTB to file  
    for result_name in ['true', 'false']:
        result_list = locals()[result_name+'_list']
        # check if filename already present
        fp = '01_production/results/' + result_name + '.csv'
        while os.path.isfile(fp):
            new_fn = input('This file \'{}\' exists. Enter a new filename to save, or \'ow\' to overwrite.'.format(result_name))
            if new_fn == 'ow': break 
            fp = '01_production/results/' + new_fn + '.csv'
        with open(fp, 'w+') as f:
            f.write('DocID, SentID, RawText, CharStartInDoc, CharEndInDoc, \n')
            for i in result_list:
                [f.write(str(i2)+', ') for i2 in i]
                f.write('\n')
            print('Results saved. Check {}.'.format(fp))

    return true_list, false_list

def make_parsedict(input_dir, fw, pdtb_v=2, lang='en'):
    """
    Takes a collection of documents, segment and tokenise each one. For each document, create a ParsePDTB instance for each of its sentence. This contains all the information necessary to 
    Parameters: 
    - input_dir (str): the path to the directory with the files holding the documents to be processed. 
    - fw (str): the syntactic framework to parse the document with, build the features and predict with the models on.
    - pdtb_v (int):  the version of the PDTB to be used (relevant for classifying the types of connectives after identification)
    - lang (str): the language of the documents being processed. 
    Return: 
    - Parse_dict (dict): a dictionary containing ParsePDTB instances. The dictionary is organised as follows: the keys correspond to the document ID, the value of each key is a list. Each list contains an ordered collection of the ParsePDTBs for the sentences in the document. 
    """
    Parse_dict = {}
    filepaths = glob.glob(input_dir+'*.txt')
    
    nlp = en_core_web_sm.load()
    if fw == 'PTB': 
        import benepar
        from benepar.spacy_plugin import BeneparComponent 
        benepar.download('benepar_en2')
        nlp.add_pipe(BeneparComponent("benepar_en2"))

    for fp in filepaths:
        # 1. open the file, load its contents 
        fn = os.path.splitext(fp)[0]
        with open(fp) as f: data = f.read().replace('\n', ' ') 
        # 2. parse with spacy
        annotated = nlp(data)
        # 3. reformat the tokenizer ouput to match the CoNLL2015 data structure. 
        wordlists = [_reformat_tokenised(tokenised_sent) for tokenised_sent in annotated.sents]
        rawtexts = [i.text for i in annotated.sents]
        if fw == 'PTB': 
            consttrees = [tokenised_sent._.parse_string for tokenised_sent in annotated.sents]
            deptrees = [None]*len(list(annotated.sents)) 
        elif fw == 'UD2': 
            consttrees = [None]*len(list(annotated.sents))
            deptrees = [_get_deptrees(tokenised_sent) for tokenised_sent in annotated.sents]
        # 4. get the PTB or UD parse 
        zipped = zip(rawtexts, wordlists, consttrees, deptrees)
        ParsePDTBs = [ParsePDTB(lang = lang, docid = fn, sentid = sent_idx, gold_consttree = None, pdtb_version = pdtb_v, 
        rawtext = rawtext, wordlist = wordlist, consttree_auto = consttree_auto, deptree_auto = deptree_auto)  \
            for sent_idx, (rawtext, wordlist, consttree_auto, deptree_auto) in enumerate(zipped)]
        Parse_dict[fn] = ParsePDTBs
    
    return Parse_dict

def _get_deptrees(tokenised_sent):
    """
    Takes a sentence parsed by spacy, extracts the relevant dependency parsing information ((1)dependency relation, (2) a token's head with its token index in the sentence, (3) the token itself with its index in sentence.) etc in order to produce representations for the sentence that are the same as those used in the CoNLL 2015 shared task. 
    Paramters:
    - tokenised_sent (spacy Span instance): a single element from spacy's Doc.sents property. Containing information about a single parsed sentence. 
    Returns: 
    - deptrees (list): a list of lists, from which a dependency parse for a sentence can be reconstructed.
    """
    deptrees = []
    for tok_idx in range(len(tokenised_sent)): 
        token = tokenised_sent[tok_idx]
        deprel, head, self_ = token.dep_, token.head.text+'-'+str(token.head.i+1), token.text+'-'+str(token.i+1)
        # ensure consistency with CONLL2015 format (matters for features later)
        if deprel == 'ROOT': deprel, head = 'root', 'ROOT-0' 
        deptrees.append([deprel, head, self_] )
    return deptrees

def _reformat_tokenised(tokenised_sent):
    """
    Takes a sentence parsed by spacy, extracts the relevant information on POS tags, character offsets etc in order to produce representations of each of its tokens that are the same as those used in the CoNLL 2015 shared task. 
    Paramters:
    - tokenised_sent (spacy Span instance): a single element from spacy's Doc.sents property. Containing information about a single parsed sentence. 
    - fw (str): the syntactic framework to be used (relevant for determining which featureset building script to use).
    Return: 
    tokenised_sent_n (list): a list of lists. Each nested list contains (1) the token, (2) a dictionary containing information about its character offsets and parts of speech tag. 
    """
    tokenised_sent_n = []
    for tok_idx, token in enumerate(tokenised_sent): 
        try: start, stop = token.idx, tokenised_sent[tok_idx+1].idx-2 #-2 because of whitespace
        except: start, stop = token.idx, token.idx+len(token.text)

        tokens = [token.text, {'CharacterOffsetBegin': start, 'CharacterOffsetEnd': stop,'PartOfSpeech': token.pos_,
        'PartOfSpeech_PTBAuto':token.pos_, 'PartOfSpeech_UDAuto': token.tag_}]
        tokenised_sent_n.append(tokens)

    return tokenised_sent_n       


if __name__ == "__main__":
    import argparse

    # 1. adding argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, default = '', required = False, 
    help='The directory path containing the files holding the documents to be processed.')
    parser.add_argument('--fw', type = str, default = 'UD2', required = False, choices = ['UD2', 'PTB'],
    help='The syntactic framework to produce the featuresets on. The appropriate trained model will be applied based on this selection.')
    parser.add_argument('--pdtb_v', type = int, default = '2', required = False, choices = [2, 3],
    help='The version of PDTB to label the Parse-class instances with. Suggestion to retain the default 2, unless planned to work with PDTB v3 labels.')
    parser.add_argument('--lang', type = str, default = 'en', required = False, choices = ['en', 'fr', 'zh'],
    help='The language that the document(s) is(are) in.')
    parser.add_argument('--connslex', type = str, default = '03_data/en/lexicons/connectives_rawmapped', 
    required = False, help='The filepath to a lexicon of connectives. This is a plaintext file containing the canonical form of a connective on a new line. This is used to (1) identify connective candidates, and (2) label them for classificaton.')
    parser.add_argument('--mapdict', type = str, default = '03_data/en/lexicons/connectives_defaultmapping', 
    required = False, help='The filepath to a json file containing a lexicon of connectives. Each key is a possible lexicalisaton of the connectives in the lexicon, and its value is the canonical form. This is used in the connective candidate identification step. ')
    args = parser.parse_args()
    
    # 2. processing, predicting and saving
    process_predict_connective(**vars(args))
    
    