# -*- coding: utf-8 -*-
#!  usr/bin/env python3
import copy, codecs, json, dill, re, time
from a1_dataloader import ParsePDTB, RelationPDTB
import a2_parsers
from nltk import ParentedTree
# for importing from sibling directories
import sys, os 
sys.path.append(os.path.abspath('../03_data'))
sys.path.append(os.path.abspath('../04_utils'))

"""
########### Notes on usage ###########
Default settings:
1. lowercase argument in get_connectivecandidates is set to True. This searches the text input for connectives in lowercase. i.e. gold explicit DCs (e.g. 'Accordingly' in the PDTB) in lowercase ('accordingly'). This effectively discards signals about the connective, especially that of being at the start of sentence. However, such signals may be obtained from other features. 
2. Note that a small number of disjoint connectives (on the one hand.. on the other hand..) present in the PDTB are not currently being handled. 
"""

def get_connectivecandidates(Parses_dict, sorted_connslex, mapping_dict, lowercase = True, posexp=None): 
    """
    Searches the RawText of a collection of Parse-class objects in order to find candidates strings that may be a discourse connective (DCs). It searches against the set of gold connectives in the PDTB that has been sorted by first token (of the gold connective), and the connective's token length. 

    This function is intended to be used to identify negative examples of DCs in 02_modelbuilding, as well as connective candidates in 01_production. 
    input |
    posexp: True, False or None - True if classified as positive example, False
    if a negative example, None if classification not performed yet. 
    output | 
    """

    ##### Find negative/positive examples of discourse connectives #####
    Relations_list = list()
    connid_counter = 0 
    for DocID in Parses_dict: 
        Parses = Parses_dict[DocID]
        Sent_counter = -1 # offset starts from 0, increment after popping
        Token_semicounter = 0
        while len(Parses) > 0:
            Parse = Parses.pop(0) # get the sentences, sequentially from start
            Sent_counter += 1
            _words_in_sent = Parse.Words # information on every word in the sentence is collected here 
            _tovisit = dict(enumerate(_words_in_sent))      # 
            _tovisit_idxs = list(_tovisit) # place keys in list 

            def __checkmake_Parseobj(_tovisit, _tovisit_idxs, sorted_connslex, Relations_list, connid_counter, posexp=posexp):
                """

                """
                while len(_tovisit_idxs) > 0: 
                    visiting_idx = _tovisit_idxs.pop(0)
                    visiting_word = _tovisit.pop(visiting_idx)[0]
                    
                    if lowercase == True: visiting_word = visiting_word.lower() 
                    else: pass

                    if visiting_word in sorted_connslex: 
                        for conn_length in sorted(sorted_connslex[visiting_word], reverse=True): # to treat longest first
                            conn_candidate_idxrange = range(visiting_idx, visiting_idx+conn_length+1) 
                            if set(conn_candidate_idxrange).issubset(set(_tovisit_idxs+[visiting_idx])): 
                                # check if there is sufficient tokens to build connective candidates

                                _slice_head, _slice_tail = conn_candidate_idxrange[0], conn_candidate_idxrange[-1] 
                                _conn_candidate_info = _words_in_sent[_slice_head:_slice_tail]
                                _conn_candidate_string = [i[0] for i in _conn_candidate_info]
                                
                                if lowercase == True: _conn_candidate_string = [i.lower() for i in _conn_candidate_string] 
                                # lowered to allow matching with gold
                                else: pass
                                
                                conns_gold = sorted_connslex[visiting_word][conn_length]
                                for conn_gold in conns_gold:
                                    if lowercase == True: conn_gold = conn_gold.lower() # to lowercase the gold to allow matching with candidate
                                    else: pass
                                    
                                    if tuple(_conn_candidate_string) == tuple(conn_gold.split(' ')): # tuples to compare content and location match
                                        _to_remove = list(conn_candidate_idxrange[:-1])
                                        _to_remove.remove(visiting_idx)
                                        # [_tovisit_idxs.remove(i) for i in _to_remove] # move this to outside the function 

                                        connid_counter += 1    
                                        connid = f'{connid_counter:04}'

                                        
                                        if posexp == False: sense, type_, arg1, arg2 = None, None, None, None # setting None values for negative examples (they will not have such attributes)
                                        else: pass
                                        # make Connective dict
                                        connective = {'RawText': None, 'CharSpanList': None, 'TokenList': None}
                                        connective['RawText'] = " ".join(_conn_candidate_string)
                                        connective['CharSpanList'] = [[min([i[1]['CharacterOffsetBegin'] for i in _conn_candidate_info]), 
                                        max([i[1]['CharacterOffsetEnd'] for i in _conn_candidate_info])    ]]

                                        # make nested TokenList 
                                        elem1 = [i[1]['CharacterOffsetBegin'] for i in _conn_candidate_info]
                                        elem2 = [i[1]['CharacterOffsetEnd'] for i in _conn_candidate_info]
                                        elem3 = [Token_semicounter+visiting_idx+i for i in range(conn_length)]
                                        elem4 = [Sent_counter]*conn_length
                                        elem5 = [visiting_idx+i for i in range(conn_length)]

                                        tokenlist = [list(i) for i in zip(elem1, elem2, elem3, elem4, elem5)]
                                        connective['TokenList'] = tokenlist

                                        # instantiate RelationPDTB object 
                                        _relation = RelationPDTB(posexp=False, connid = connid, connective = connective, 
                                        sense = sense, type_ = type_, arg1 = arg1, arg2 = arg2, 
                                        pdtb_version=Parse.PDTBVersion, lang = Parse.Lang, docid = Parse.DocID, 
                                        sentid = Sent_counter)

                                        # map the connective 
                                        _relation._map_connective(mapping_dict)
                                        Relations_list.append(_relation)
                                    else: 
                                        _to_remove = []
                            else: 
                                _to_remove = []
                    else: 
                        _to_remove = []
                                        
                return Relations_list, _to_remove

            Relations_list, _to_remove = __checkmake_Parseobj(_tovisit, _tovisit_idxs , sorted_connslex, Relations_list, connid_counter)
            _to_remove_copy = _to_remove.copy() # copy to ensure we don't change _to_remove

            if len(_to_remove_copy) >1: 
                _words_in_subsent = _words_in_sent[min(_to_remove_copy), max(_to_remove_copy)+1] # plus one for slicing so as to include last elem of _to_remove_copy
                _tovisit = dict(enumerate(_words_in_subsent))      # 
                _tovisit_idxs = list(_tovisit) # place keys in list
                Relations_list, _to_remove_copy = __checkmake_Parseobj(_tovisit, _tovisit_idxs, sorted_connslex, Relations_list, connid_counter)
            else: pass 

            [_tovisit_idxs.remove(i) for i in _to_remove]
            
            continue
        Token_semicounter += len(Parse.Words)
    return Relations_list

def __check_sentid(tokenlist, sentid_idx = -2):
    """
    helper function, used whenever SentID needs to be retrieved. Checks that, in
    the TokenList provided, the sentence offset across all Tokens is the same.
    Intended use is on the Connective TokenList, to check that the connective
    appears within the same sentence (i.e. not disjoint/parallel connective)
    """
    _SentID_set = set([i[sentid_idx] for i in tokenlist])
    try:                 # check that there is only 1 unique sentence no.
        assert len(_SentID_set) == 1
        return _SentID_set.pop()
    except AssertionError as e:
        e.args += ('Connective candidate appears to be across multiple sentences')
    raise

def __make_sorted_connslex(connslex):
    """
    Helper function to sort the canonical form of connectives by their first token (since connectives may be multi-word expressions), as well as well as the length of the entire connective. 
    """

    # 1. create dictionary of dictionaries, containing lists. top-level key is first token of connective, second-level key is # tokens in connective. 
    sorted_connslex = {conn.split()[0].lower(): {} for conn in connslex}
    [sorted_connslex[conn.split()[0].lower()].update({len(conn.split()):set()}) for conn in connslex] 
    # populate dictionary using list_comp
    [sorted_connslex[conn.split()[0].lower()][conn_length].add(conn.lower())  \
    for conn in connslex \
    for conn_length in sorted_connslex[conn.split()[0].lower()] \
    if len(conn.split()) == conn_length]

    return sorted_connslex

if __name__ == "__main__":
    # input the language code for dataset to be processed
    LANG = input('Enter the iso code for the language to process; \'en\', \'fr\', \'zh\', are currently available. \n')
    PDTB_VERSION = 2

    ##### 1. Load data #####
    # a. run _sorted_connslex and pass it on 
    # using the mapped set of connectives (i.e. from the ConnHeadMapper.py provided by the Shared Task organisers) instead of the raw set. 
    with open('./03_data/{}/lexicons/connectives_rawmapped'.format(LANG)) as f:
        connslex = [i.strip('\n') for i in f.readlines()]    
    sorted_connslex = __make_sorted_connslex(connslex) 

    # b. load the PDTB explicit DC mappings. in PDTB 2.0 manual and provided by
    # CoNLL2015 Shared Task organisers
    # https://github.com/attapol/conll15st/blob/master/conn_head_mapper.py 
    with open('./03_data/{}/lexicons/connectives_defaultmapping'.format(LANG)) as f:  
        expconn_mapping_dict = json.load(f)

    ##### 2. loading the CoNLL 2016 train, dev and test sets         #####
    # a.  rawtext, syntactic parse etc information 
    for dataset in  ['train','dev', 'test']:
        globals()['parsefile_'+dataset] = "./03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/parses.json".format(LANG, dataset)
    
        with open(globals()['parsefile_'+dataset]) as f:
            globals()['parse_'+dataset] = json.load(f)

    # b. discourse relations information 
        globals()['pdtb_file_'+dataset] = codecs.open("./03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/relations.json".format(LANG,dataset), encoding='utf8')
    
        globals()['relations_'+dataset] = [json.loads(x) for x in globals()['pdtb_file_'+dataset]]


    ##### 3. Generate and populate the ParsePDTB RelationPDTB classes      #####
    # a. The code segment below loads all the sentences in section 02 to 22 of
    # the PDTB2.0 

    
    def __obtain_ConstTrees_Gold(DocID, readpath = './03_data/{}/{}tbRoot/{}/', lang = LANG):
        """
        Helper function to retrieve and pre-process treebank gold constituency parses from .mrg (PTB)/.pid (CTB)/.txt (FTB) files. 
        """
        if lang == 'en':
            _folder = DocID.split('_')[-1][0:2]
            readpath = readpath.format(LANG, 'P', _folder)
            with open(readpath+DocID + '.mrg', 'r') as f:
                doc = f.readlines()
            # strip lines that only consists of newline 
            doc = [i.lstrip().replace('\n', '') for i in doc if i != '\n']

            _ConstTrees = list()
            open_brac = 0
            while len(doc) > 0: 
                _curr = doc.pop(0)
                open_brac += (len(re.findall( r'\(', _curr)) - len(re.findall( r'\)', _curr))) 
                _1ConstTree = [_curr]
                while open_brac > 0:
                    _curr = doc.pop(0)
                    open_brac += (len(re.findall( r'\(', _curr)) - len(re.findall( r'\)', _curr))) 
                    _1ConstTree.append(_curr)

                _ConstTrees.append(_1ConstTree)
                open_brac = 0
                continue
            ConstTrees = [''.join(i) for i in _ConstTrees]
        elif lang == 'fr': pass
        elif lang == 'zh': pass 
        else: raise ValueError
        
        return ConstTrees
         
    def __populate_Parses(lang, parsejson, new_parsedict): 
        """
        """
        # start CoreNLP servers for UD1 
        from stanfordnlp.server import CoreNLPClient

        cwd =  os.getcwd()
        version = 'stanford-corenlp-full-2018-10-05'
        corenlp_path = re.findall(r'\S*/marta-v2', cwd)[0] + '/04_utils/' + version
        os.environ["CORENLP_HOME"] = corenlp_path
        if lang == 'en': 
            lang = {} # i.e. CoreNLP defaults to English model
            corenlpclient_UD1 = CoreNLPClient(properties = {'ssplit.isOneSentence': True, 'tokenize.whitespace': True	}, annotators = ['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'udfeats'], memory='2G', be_quiet=True, max_char_length=100000, output_format = 'conllu') 
            # parse annotator is necessary to obtain udfeats (for postags)

        if lang == 'fr': 
            lang = 'french'
            corenlpclient_UD1 = CoreNLPClient(properties = lang, annotators = ['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'udfeats'], memory='2G', be_quiet=True, max_char_length=100000, output_format = 'conllu') # note that udfeats (for postags) currently works for english only https://stanfordnlp.github.io/CoreNLP/udfeats.html

        if lang == 'zh': 
            lang = 'chinese'
            corenlpclient_UD1 = CoreNLPClient(properties = lang, annotators = ['tokenize', 'ssplit', 'pos', 'parse', 'depparse', 'udfeats'], memory='2G', be_quiet=True, max_char_length=100000, output_format = 'conllu')
            # note that udfeats (for postags) currently works for english only https://stanfordnlp.github.io/CoreNLP/udfeats.html

        # begin processing
        for DocID in parsejson: 
            print('Now processing: ', dataset, DocID)
            sentence_offset = 0 # this is the 4th element in a TokenList
            
            # obtain the gold constituency parses for the document. 
            ConstTrees = __obtain_ConstTrees_Gold(DocID, readpath = './03_data/{}/{}tbRoot/{}/', lang = LANG)

            for sentence in parsejson[DocID]['sentences']:
                # 1. create a ParsePDTB object 
                __parsepdtb = ParsePDTB(lang = LANG, docid = DocID, sentid = sentence_offset, gold_consttree = ConstTrees[sentence_offset],pdtb_version = PDTB_VERSION)
                 
                # 2. add to .RawText and .Words
                __parsepdtb.RawText = " ".join([word[0] for word in sentence['words']])
                __parsepdtb.Words = sentence['words']
                
                # 3. add to ConstTree_Auto. generate parse if missing 
                if sentence['parsetree'] == '(())\n':
                    _parse = a2_parsers._parse_rawtext2consttree(LANG, __parsepdtb.RawText, tokenized=True)
                    __parsepdtb.ConstTree_Auto = _parse
                else: __parsepdtb.ConstTree_Auto = sentence['parsetree']
                
                # 3. write to temp file, for converting to SD/UD1 in next steps
                with open('./02_modelbuilding/02_output/input_temp.parser', 'w+') as f: 
                    f.write(__parsepdtb.ConstTree_Gold)

                # 4. convert constituency parse to gold UD 1.0 and add to DepTree_UD1_Gold
                a2_parsers.convert_const2dep(LANG, dataset, filename = '', readpath = '/02_modelbuilding/02_output/input_temp.parser', writepath ='/02_modelbuilding/02_output/output_temp.parser',format_ = 'UD1', usage='experiments')

                with open('./02_modelbuilding/02_output/output_temp.parser', 'r') as f: 
                    UD1_Gold_conllu = f.read()
                def __conllu2tuple(conllu_doc):
                    """helper function to convert CoNLL format into 3-tuple used by CoNLL 2016 organisers to store dependency parses
                    """
                    to_list = conllu_doc.split('\n')
                    tokenlist = [i.split('\t')[1]+'-'+i.split('\t')[0] for i in to_list if i != ''] # convert  CoNLL line to <wordform>-<token num>
                    tokenlist.insert(0, 'ROOT-0') # add a root token to the start 
                    deptree_gold = [[i.split('\t')[7], tokenlist[int(i.split('\t')[6])], i.split('\t')[1]+'-'+i.split('\t')[0]] for i in to_list if i !=''] # convert to CoNLL 2016 dependencies format
                    return deptree_gold
                __parsepdtb.DepTree_UD1_Gold = __conllu2tuple(UD1_Gold_conllu)
                
                # 5. automatically generate UD 1.0 constituency parse (from raw text), place into same 3-tuple format as CoNLL 2016 Shared Task,and add to DepTree_UD1_Auto
                UD1_Auto_conllu = corenlpclient_UD1.annotate(__parsepdtb.RawText)
                __parsepdtb.DepTree_UD1_Auto =  __conllu2tuple(UD1_Auto_conllu) 

                # 6. add PTB-style and UD pos tags to .Words. Each of the variable below contain a list comprising 2-tuples. each tuple is (<wordform>, <part of speech>)

                globals()['pos_PTBGold'] = [i for i in ParentedTree.fromstring(__parsepdtb.ConstTree_Gold).pos() if i[-1]!='-NONE-'] # gold PTB parses have traces and these causes misalignment with the surface form. we drop these since parsers don't predict traces (Johannsen & Søgaard, 2013)
                globals()['pos_PTBAuto'] = ParentedTree.fromstring(__parsepdtb.ConstTree_Auto).pos()
                globals()['pos_UDGold'] = [(i.split('\t')[1],i.split('\t')[3]) for i in UD1_Gold_conllu.split('\n') if i != '']                
                globals()['pos_UDAuto'] = [(i.split('\t')[1],i.split('\t')[3]) for i in UD1_Auto_conllu.split('\n') if i != '']    


                for postagset in ['PTBGold', 'PTBAuto', 'UDGold', 'UDAuto']: 
                    try: 
                        _tagset = globals()['pos_' + postagset]
                        assert len(_tagset) == len(__parsepdtb.Words)
                        for idx in range(len(__parsepdtb.Words)):
                            # add the part of speech as a new key in the dictionary for the token in .Words
                            __parsepdtb.Words[idx][1].update({'PartOfSpeech_'+ postagset : _tagset[idx][1]})

                    except AssertionError as e: 
    
                        e.args += (postagset.upper() + " is not of the same size as the .Words attribute for this sentence.",)
                        print(e)
                        print("Continuing to attempt alignment of tokens.")
                        _words = [i[0] for i in __parsepdtb.Words]
                        _words_maxidx = len(_words)-1

                        #'drop' the additional tokens in _tagset   
                        _tagset = [i for i in _tagset if i[0] in _words]
                        _words_curridx = -1 # start with -1 
                        for idx in range(len(_tagset)):
                            _words_curridx +=1
                            while __parsepdtb.Words[_words_curridx][0] != _tagset[idx][0] and _words_curridx < _words_maxidx: 
                                __parsepdtb.Words[_words_curridx][1].update({'PartOfSpeech_'+ postagset : 'ParserError'}) # place a marker identifying the missing pos tag as an error from parsing 
                                _words_curridx +=1 
                            __parsepdtb.Words[_words_curridx][1].update({'PartOfSpeech_'+ postagset : _tagset[idx][1]})
                            continue
                        # raise
                sentence_offset += 1 # increase sentence offset before moving to handle next sentence

                try:
                    new_parsedict[DocID].append(__parsepdtb)
                except: 
                    new_parsedict[DocID] = [__parsepdtb]

        # shut down the CoreNLP servers 
        corenlpclient_UD1.stop()

    # b. Generate RelationPDTB for positive examples of ExpConns, as well as add
    # to the ParsePDTB instances

    def __populate_Relations(relationsjson, new_relationlist, new_parsedict): 
        """
        """
        for relation in relationsjson: 
            if relation['Type'] == 'Explicit':
                
                sentid = __check_sentid(relation['Connective']['TokenList'], sentid_idx = -2)

                _relationpdtb = RelationPDTB(posexp = True, connid = relation['ID'], connective = relation['Connective'],
                                            sense = relation['Sense'], type_ = relation['Type'],
                                            arg1 = relation['Arg1'], arg2 = relation['Arg2'],
                                            pdtb_version = 2, lang = LANG, 
                                            docid = relation['DocID'], sentid = sentid)
                _relationpdtb._map_connective(expconn_mapping_dict)     # add the connective mapping
                new_relationlist.append(_relationpdtb)
                
                # for populating ParsePDTB instance 
                _conninfo   = _relationpdtb.Connective
                _SentID     = _relationpdtb.Connective['TokenList'][0][-2] 
                _DocID       = _relationpdtb.DocID
                _conninfo['ConnID'] = _relationpdtb.ConnID              # this is the unique identifer for the  
                                                                        # connective. It helps in linking arguments
                _conninfo['Sense'] = _relationpdtb.Sense
                _Type = _relationpdtb.Type 
                
                # store to the relevant ParsePDTB instance 
                new_parsedict[_DocID][_SentID].Connectives[_Type].append(_conninfo)
        print("number: Relations_ExpConn_posexps_list", len(new_relationlist))

    # c. identify negative examples and generate RelationPDTB instances
    ##### Pre-treatment #####
    def _pretreat_negativeexamples(new_relationlist, newparse_dict): 
        """

        """
        newparse_dict_copy = copy.deepcopy(newparse_dict) # make a deep copy of the Parses_list
        for Relation in new_relationlist: # iterate through all posexp Relations 
            _DocID = Relation.DocID  # ID no. for where the connective is in 
            _tokenslist = Relation.Connective['TokenList'] # extract the connective's TokenList
            _to_overwrite = [i[-1] for i in _tokenslist] # idx of tokens to overwrite in sentence
            _SentID = __check_sentid(_tokenslist) # get the SentID from the TokenList
            
            # set overwrite_char in the relevant position in each Parse 
            _Parse = newparse_dict_copy[_DocID][_SentID]
            _RawTextTokens = _Parse.RawText.split()
            for overwrite_idx in _to_overwrite:
                _RawTextTokens[overwrite_idx] = '\u1300'    # overwrite_char defaults to Ethopian ጀ, 
                _Parse.Words[overwrite_idx][0] = '\u1300'   # which is not expected to be typically encountered in the languages we plan to handle.  
        return newparse_dict_copy

    # Run the three functions above on the train, dev, test sets
    for dataset in ['train', 'dev', 'test']: 
        # a. run the __populate_Parses function on the train, dev, test sets
        globals()['ParsePDTB_dict_'+dataset] = dict()
        __populate_Parses(lang = LANG, parsejson = globals()['parse_'+dataset], new_parsedict = globals()['ParsePDTB_dict_'+dataset])

        # b. run the __populate_Relations function on the train, dev, test sets
        globals()['Relations_ExpConn_posexps_'+dataset] = list()
        __populate_Relations(globals()['relations_'+dataset], globals()['Relations_ExpConn_posexps_'+dataset], globals()['ParsePDTB_dict_'+dataset])

        # c. run _pretreat_negativeexamples and then get_connectivecandidates on the returned results to obtain the negative examples
        globals()['ParsePDTB_dict_'+dataset+"_copy"] = _pretreat_negativeexamples(globals()['Relations_ExpConn_posexps_'+dataset], globals()['ParsePDTB_dict_'+dataset])
        
        globals()['Relations_ExpConn_negexps_'+dataset] = get_connectivecandidates(globals()['ParsePDTB_dict_'+dataset+'_copy'], sorted_connslex, mapping_dict = expconn_mapping_dict, lowercase = True, posexp=False)

        print(dataset.upper()+' ExpConn: Positive examples: ', len(globals()['Relations_ExpConn_posexps_'+dataset]), 
        dataset.upper()+' ExpConn: Negative examples: ', len(globals()['Relations_ExpConn_negexps_'+dataset]))


        # save Relations_ExpConn_posexps_list and Relations_ExpConn_negexps_list to dill files
        with open("03_data/{}/explicit_connectives/ExpConn_posexp_{}.dill".format(LANG,dataset), "wb") as f:
            dill.dump(globals()['Relations_ExpConn_posexps_'+dataset], f)
        with open("03_data/{}/explicit_connectives/ExpConn_negexp_{}.dill".format(LANG, dataset), "wb") as f:
            dill.dump(globals()['Relations_ExpConn_negexps_'+dataset], f)
        with open("03_data/{}/explicit_connectives/README_{}set.md".format(LANG,dataset), "w+") as f:
            pos_count = len(globals()['Relations_ExpConn_posexps_'+dataset])
            neg_count = len(globals()['Relations_ExpConn_negexps_'+dataset])
            f.write('There are {} positive examples and {} negative examples in the {} set.'.format(str(pos_count), str(neg_count), dataset))
        # save ParsePDTB_dict to dill files 
        with open("03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill".format(LANG,dataset), "wb") as f:
            dill.dump(globals()['ParsePDTB_dict_'+dataset], f)
        

    LANG = 'en'
    dataset='train'
    with open("03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill".format(LANG,dataset), "rb") as f:
        conns = dill.load(f)
    
    print(conns['wsj_1057'][143].RawText)
    print(conns['wsj_1057'][142].ConstTree_Auto)
    print(conns['wsj_1057'][142].ConstTree_Gold)
    print(conns['wsj_1057'][142].DepTree_UD1_Auto)
    print(conns['wsj_1057'][142].DepTree_UD1_Gold)



##### Changes made:
# current: v1.3
# 1. added assertions to ensure number of features in dev and test matches train
# 2. identification of negative examples include possible DCs within matches for longer MWE DCs (e.g. "when and if") 
# previous: v1.1 
# 1. add check for connective type in code for positive examples 
# 2. added lowercase option (in get_connectivecandidates) for the 354 gold connectives, sorted_connlex becomes a set 
# 3. generalised the get_negexp function to get_connectivecandidates, this will be used in the parsing pipeline to identify candidate connectives to be classified. 
# previous: v1.2
# 1. nested from stanfordnlp.server import CoreNLPClient into the __populate_Parses function. This is only required for producing parses for the experiments and not for production. Avoid production users having to install stanfordnlp (which comes with Torch dependencies). 