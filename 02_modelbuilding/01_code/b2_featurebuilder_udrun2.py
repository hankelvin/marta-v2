# -*- coding: utf-8 -*-
from operator import itemgetter 
import b1_featurebuilder_ptbrun1

"""
##### Notes on use                                                         #####
1. This module utilises two different index formats for obtaining the feature values. 
 a. The first is the WordToken indices used in the CoNLL 2015 Shared Task. For each word, this is a fixed-sized (5-element) tuple. 
2. The naming convention for the functions the feature classes corresponds to the feature names (e.g. part of speech for connective from Lin et al 2010, CPOS), so they can be easily extracted from a module inspection in b_featurebuilder.py. 
3. Note that the _idx output from each function here is different from those output in the PTB-style set of functions. In any case, these _idx outputs are not used elsewhere and are merely a check. 

"""

######### Functions to build features in connective identification ##########
##### Dependency tree-based features                                   #####
##### from Pitler & Nenkova 2009                                        #####

class PitlerNenkova_Conn_UD1: 
    def CStr(*args, **kwargs): # input: Relation_obj)
        """
        Returns the string form of the connective candidate (e.g. 'and', 'or', etc). 
        input | Relation_obj: Relation-class object - object containing information about a single discourse relation. 
        output | str - the wordform(s) in the discourse connective (MappedText or RawText)
        """
        _label = itemgetter('connstring')(args[0])
        _idx = itemgetter('connidx')(args[0])

        return (_label, _idx)
        
    
# class PitlerNenkova_Syn_UD1:

    # the following features are removed in Run 2. There is no direct equivalent in the UD framework. 
    # SELF = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.SELF
    # PRNT = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.PRNT 
    # LEFTSib = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.LEFTSib
    # RGHTSib = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.RGHTSib


##### from Lin et al 2010                                               #####

class Lin_etal_UD1: 

    def CPOS(*args, **kwargs):
        """
        Returns the part of speech of the connective candidate. 
        """
        connidx, wordslist, gold_ = itemgetter('connidx', 'wordslist', 'gold_')(args[0])
        if len(connidx) == 1: 
            if gold_ == True: CPOS = wordslist[connidx[0]][1]['PartOfSpeech_UDGold']
            if gold_ == False: CPOS = wordslist[connidx[0]][1]['PartOfSpeech_UDAuto']
            _label = CPOS
            _idx = connidx
        else: 
            if gold_ == True: CPOS = [wordslist[i][1]['PartOfSpeech_UDGold'] for i in connidx]
            if gold_ == False: CPOS = [wordslist[i][1]['PartOfSpeech_UDAuto'] for i in connidx]
            _label = '_'.join(set(CPOS)) # placing in a set to minimise sparsity
            _idx = connidx

        return (_label, _idx)

    def Prev_C(*args, **kwargs):
        """
        Returns the string form of the word before the connective. (corresponds to the prev1 + C string feature in Lin et al). For MWE connectives, this is the string immediately to the left of the first word of the connective. 
        """
        connstring, connidx, deptree = itemgetter('connstring', 'connidx', 'deptree')(args[0])

        if min(connidx) <1:
            _idx = "No_idx"  
            prevstring = "NoPrevWord"
        else: 
            _idx = min(connidx)-1
            prevstring = deptree[_idx][2].split('-')[0]
        
        _idx = ["No_idx"] # to align with outputs of every func in this module
        _label = prevstring+ '^' +connstring
        
        return (_label, _idx)
        
    def PrevPOS(*args, **kwargs):
        """
        Returns a string that is a concatenation of the parts of speech for the word before the connective and for the connective itself. (corresponds to the prev1 POS + C POS feature in Lin et al) 

        """
        connidx, wordslist, gold_ = itemgetter('connidx', 'wordslist', 'gold_')(args[0])
        
        if min(connidx) <1:
            _idx = "No_idx"  
            prevPOS = "NoPrevPOS"
        else: 
            _idx = min(connidx)-1
            if gold_ == True: prevPOS = wordslist[_idx][1]['PartOfSpeech_UDGold']
            if gold_ == False: prevPOS = wordslist[_idx][1]['PartOfSpeech_UDAuto']
        
        _idx = [_idx] # to align with outputs of every func in this module
        _label = prevPOS
        
        return (_label, _idx)
        
    def PrevPOS_CPOS(*args, **kwargs):
        """
        Returns the part of speechs of the word coming before the connective candidate, together with that of the connective itself. 
        """
        # the inputs to *args are nested in a tuple. args[0] extracts the inputs.in this case it is initvals_dict, which we want to pass into the 2 previously written functions  PrevPOS and CPOS
        prevPOS = Lin_etal_UD1.PrevPOS(args[0])[0]
        CPOS = Lin_etal_UD1.CPOS(args[0])[0]
        _idx = ['No_idx'] # to align with outputs of every func in this module
        _label = prevPOS+ '^' +CPOS

        return (_label, _idx)

    def C_Next(*args, **kwargs):
        """
        Returns the string form of the word immediately after the connective.(corresponds to the C string + next1  feature in Lin et al) For MWE connectives, this is the string immediately to the right of the last word of the connective. 
        """
        connstring, connidx, deptree, sentlength = itemgetter('connstring', 'connidx', 'deptree', 'sentlength')(args[0])

        if max(connidx) == sentlength: # note max(connidx) should not be > than sentlength, if it is there must be an issue somewhere else.
            _idx = "No_idx"  
            nextstring = "NoNextWord"
        else: 
            _idx = max(connidx)+1
            nextstring = deptree[_idx][2].split('-')[0]
        
        _idx = ['No_idx'] # to align with outputs of every func in this module
        _label = connstring+ '^' +nextstring
        
        return (_label, _idx)

    def NextPOS(*args, **kwargs):
        """
        Returns the part of speech for the word immediately after the connective. (corresponds to the next1 POS feature in Lin et al)  
        """
        connidx, wordslist, sentlength, gold_ = itemgetter('connidx', 'wordslist', 'sentlength', 'gold_')(args[0])
        
        if max(connidx) == sentlength: # note max(connidx) should not be > than sentlength, if it is there must be an issue somewhere else.
            _idx = "No_idx"  
            nextPOS = "NoNextPOS"
        else: 
            _idx = max(connidx)+1
            if gold_ == True: nextPOS = wordslist[_idx][1]['PartOfSpeech_UDGold']
            if gold_ == False: nextPOS = wordslist[_idx][1]['PartOfSpeech_UDAuto']
        
        _idx =[_idx] # to align with outputs of every func in this module
        _label = nextPOS
        
        return (_label, _idx)

    def CPOS_NextPOS(*args, **kwargs):
        """
        Returns the parts of speech of the connective candidate, together with that of the word coming after the connective candidate. 
        """
        # the inputs to *args are nested in a tuple. args[0] extracts the inputs.in this case it is initvals_dict, which we want to pass into the 2 previously written functions  PrevPOS and CPOS
        NextPOS = Lin_etal_UD1.NextPOS(args[0])[0]
        CPOS = Lin_etal_UD1.CPOS(args[0])[0]
        _idx = ['No_idx'] # to align with outputs of every func in this module
        _label = CPOS+ '^' +NextPOS

        return (_label, _idx) 
     
    # the following features present in Run 1 are removed in Run 2. This is because there is no direct equivalent in the UD framework. 
    # FullPath = b1_featurebuilder_ptbrun1.Lin_etal.FullPath 
    # CompPath = b1_featurebuilder_ptbrun1.Lin_etal.CompPath

##### Li et al 2016                                                #####

class Li_etal16_UD1:
    """
    The set of features used by Li et al in their submission for the CoNLL 2016 Shared Task. 
    """
    Prev_C = Lin_etal_UD1.Prev_C
    PrevPOS = Lin_etal_UD1.PrevPOS
    PrevPOS_CPOS = Lin_etal_UD1.PrevPOS_CPOS
    C_Next = Lin_etal_UD1.C_Next
    NextPOS = Lin_etal_UD1.NextPOS
    CPOS_NextPOS = Lin_etal_UD1.CPOS_NextPOS
    # the following features present in Run 1 are removed in Run 2. This is because there is no direct equivalent in the UD framework. 
    # SELF = PitlerNenkova_Syn.SELF

########## Helper funcs to get indices, navigate between collections ##########
########## of  Parse-class and Relation-class objects                ##########

_CStr = b1_featurebuilder_ptbrun1._CStr
_get_connidx = b1_featurebuilder_ptbrun1._get_connidx

def _retrieve_wordslist(Relation_obj, Parse_dict):
    """
    Return the tokenised words list of the sentence. This contains information about the token's index in the document and sentence, as well as its part of speech tags (in PTB as well as UD1 tags). 
    """
    DocID, SentID = Relation_obj.DocID, Relation_obj.SentID
    wordslist = Parse_dict[DocID][SentID].Words

    return wordslist 

def _retrieve_deptree(Relation_obj, Parse_dict, gold):
    """
    Return the dependency parse of the sentence (in UD1 3-tuples)
    """
    DocID, SentID = Relation_obj.DocID, Relation_obj.SentID
    if gold == True: 
        # try-except for legacy dataset reasons
        try: deptree = Parse_dict[DocID][SentID].DepTree_UD1_Gold
        except AttributeError: Parse_dict[DocID][SentID].DepTree_Gold
    if gold == False: 
        try: deptree = Parse_dict[DocID][SentID].DepTree_UD1_Auto
        except: deptree = Parse_dict[DocID][SentID].DepTree_Auto

    return deptree

def _make_initvars(Relation_obj, Parse_dict, gold): 
    """
    Generate the variables and values necessary to initialise all the functions in this module.  
    """
    connstring = _CStr(Relation_obj)[0]
    connidx = _get_connidx(Relation_obj)
    deptree = _retrieve_deptree(Relation_obj, Parse_dict, gold)
    wordslist = _retrieve_wordslist(Relation_obj, Parse_dict)
    sentlength = len(wordslist)
    gold_ = gold 

    return connstring, connidx, deptree, wordslist, sentlength, gold_


if __name__ == "__main__":
    # script tests below 
    import dill 
    with open('03_data/en/explicit_connectives/ExpConn_posexp_dev.dill', 'rb') as f:
        Relation_dict = dill.load(f)
        Relation_obj = Relation_dict[0]
    with open('03_data/en/pdtb_conll_data/ParsePDTB_dict_dev.dill', 'rb') as f:
        Parse_dict = dill.load(f)

    initvars = ['connstring', 'connidx', 'deptree', 'wordslist', 'sentlength', 'gold_' ]
    initvals = _make_initvars(Relation_obj, Parse_dict, gold=True)
    print(initvals)

    initvars_dict = {k:v for k,v in zip(initvars, initvals)}
    print('CStr', PitlerNenkova_Conn_UD1.CStr(initvars_dict))
    print('CPOS', Lin_etal_UD1.CPOS(initvars_dict))
    print('C_Next', Lin_etal_UD1.C_Next(initvars_dict))
    print('CPOS_NextPOS', Lin_etal_UD1.CPOS_NextPOS(initvars_dict))
    print('NextPOS', Lin_etal_UD1.NextPOS(initvars_dict))
    print('Prev_C', Lin_etal_UD1.Prev_C(initvars_dict))
    print('PrevPOS', Lin_etal_UD1.PrevPOS(initvars_dict))
    print('PrevPOS_CPOS', Lin_etal_UD1.PrevPOS_CPOS(initvars_dict))

    pass