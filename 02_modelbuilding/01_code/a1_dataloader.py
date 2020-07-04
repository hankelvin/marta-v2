# -*- coding: utf-8 -*-
import codecs, json, dill 

########## Classes to store discourse relations and parse information ##########
# This script was designed around loading the CoNLL 2015 Shared Task datasets
# see https://www.cs.brandeis.edu/~clp/conll15st/ It has been designed to
# allow expansion into  handling multilingual PDTB (and other discourse frameworks), 

##### Changes made:
# current: v1.2
# from v1.0 to v1.1: 
# 1. Replaced Connective-class in a_preprocessor to Relation class here 
# from v1.1 to v1.2:
# 1. removed ParsePDTB inheritance, added optional arguments (needed during production usage)
    
class ParsePDTB:  
    """
    Class to store  additional attributes unique to the Penn Discourse Tree Bank dataset. 
    """
    def __init__(self, lang, docid, sentid, gold_consttree, pdtb_version, rawtext = None, 
    wordlist = None, consttree_auto = None, deptree_auto = None):
        self.RawText = rawtext              # str: the raw text of the sentence.
        self.Lang = lang                    # str: the 2-character ISO code for the language of the document/sentence
        self.DocID = docid                  # str: unique identifier for document that this sentence is from.
        self.SentID = sentid                # str: unique (sequential) sentence number within the document.
        if wordlist: self.Words = wordlist  # list: of lists. each list containing information about a token in the sentence
        if gold_consttree: self.ConstTree_Gold =  gold_consttree 
                                            # str: the gold-label constituency parse for the sentence (used in exp stage)
        self.ConstTree_Auto = consttree_auto # str: the automatically-generated constituency parse for the sentence
        self.DepTree_Auto = deptree_auto    # list: of lists. each containing the deprel and the head and self tokens plus idx
        self.PDTBVersion = pdtb_version     # int: the PDTB version number that the data is from.  

        if self.PDTBVersion == 2: 
            self.Connectives = dict( {'Explicit':list(), 'Implicit': list(), 'AltLex': list(),'NoRel': list()})  
        elif self.PDTBVersion == 3:
            self.Connectives = dict({'Explicit':list(), 'Implicit': list(), 'AltLex': list(),'NoRel': list(),
                                    'Hypophora': list(), 'AltLexC': list()}) 

    def __str__(self):
        return [connective.RawText for connective in self.Connectives.values()]

class RelationPDTB():
    """
    Class to hold instances of discourse relations in PDTB. 

    The information is this class parallels some of the information in Parse, adds to them, and is intended to be mostly self-contained to facilitate the building of features and representations for each spand of related segments. Indexical information about the discourse relation is contained in this class. However, other information necessary (syntactic parses) for the building of features are found in the Parse class.
    """
    def __init__(self, posexp, connid, connective, sense, type_, 
                arg1, arg2, pdtb_version, lang, docid, sentid,  *args, **kwargs):
        self.PosExp = posexp        # ternary: True if this is a positive example of connective, False if not and None if not   ascertained yet. 
        self.ConnID = connid        # str: a unique identifier for this connective.
        self.Connective = connective# dict: containing the CharacterSpanList, RawText and TokenLists of the connective. 
        self.Arg1 = arg1            # dict: containing the CharacterSpanList, RawText and TokenLists of arg1.
        self.Arg2 = arg2            # dict: containing the CharacterSpanList, RawText and TokenLists of arg2. 
        self.Sense = sense          # list: each element a string, each element a PDTB sense that the connective has.  
        self.Type = type_           # str: the connective type (e.g. Explicit, Implicit ...).
        self.PDTBVersion = pdtb_version
        self.Lang = lang
        self.DocID = docid          # str: unique identifier for document that this sentence is from
        self.SentID = sentid        # str: unique (sequential) sentence number within the document   


    def _map_connective(self, mapping_dict):
        """
        Adds a MappedText key-value pair to the self.Connective attribute. Variants of Explicit DCs are mapped to a common one between them, e.g. "18 months after" and "25 years later" map to "after". Used in the get_connectivecandidates method in a1_dataloader. 
        """
        if self.Connective['RawText'] in mapping_dict:
            self.Connective["MappedText"] = mapping_dict[self.Connective['RawText']]
        else: 
            self.Connective["MappedText"] = self.Connective['RawText']
        pass 

    def __str__(self):
        return self.Connective['RawText']


if __name__ == "__main__":

    pass