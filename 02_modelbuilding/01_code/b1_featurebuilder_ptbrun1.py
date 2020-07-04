# -*- coding: utf-8 -*-
from nltk import tree 
from operator import itemgetter 

"""
##### Notes on use                                                         #####
1. This module relies on the nltk.tree package https://www.nltk.org/_modules/nltk/tree.html. for handling 
PTB-style constituency parsing. 
2. This module utilises two different index formats for obtaining the feature values. 
 a. The first is the WordToken indices used in the CoNLL 2015 Shared Task. For each word, this is a fixed-sized 
 (5-element) tuple. 
 b. The second is the treeposition indices found in the NLTK.tree.ParentTree class. It is a list of variable 
 length providing the position of the word within the tree for the constituency parse tree. 
3. The naming convention for the functions the feature classes corresponds to the feature names (e.g. part of 
speech for connective from Lin et al 2010, CPOS), so they can be easily extracted from a module inspection in
 b_featurebuilder.py. 

"""

######### Functions to build features in connective identification ##########
##### Constituency parse-based features                                 #####
##### from Pitler & Nenkova 2009                                        #####

class PitlerNenkova_Conn: 
    def CStr(*args, **kwargs): 
        """
        Returns the string form of the connective (e.g. 'and', 'or', etc). 
        input | Relation_obj: Relation-class object - object containing information about a single discourse relation. 
        output | str - the wordform(s) in the discourse connective (MappedText or RawText)
        """
        _label = itemgetter('connstring')(args[0])

        return (_label, "No_idx")

class PitlerNenkova_Syn:
    def SELF(*args, **kwargs): # input: connidx = connidx, nltkptree = nltkptree):
        """
        Returns the syntactic category of an explicit connective. For multi-word DCs, this is:'The highest 
        node in the tree which dominates the words in the connective but nothing else.' 

        input | connidx: list - output from _get_connidx, nltkptree: nltk.ParentTree object, features: dict - 
        dictionary with feature values and associated label
        output | _label: int - the mapped label for the feature value, _idx: tuple - the treeposition indices
        """
        # find the leaf node(s) corresponding to the token id(s). This will be unannotated word(s). 

        connidx, nltkptree = itemgetter('connidx', 'nltkptree')(args[0])

        if len(connidx) == 1: 
            # case 1: single word DC    
            _leaf_treeposition = list(nltkptree.treeposition_spanning_leaves(min(connidx), max(connidx)+1)) 
            # +1 as treeposition_spanning_leaves end is not included in search. 
            
            # in some cases, a wordform may have its own syntactic category. for e.g. : 
            # (S (CC But) (NP (NNS analysts)) (VP (VBP reckon)... so... pop the last element in order to 
            # get the treeposition for the highest node that dominates the leaf node (this will be the 
            # wordform's part of speech, its lowest syntactic category)
            while len(nltkptree[_leaf_treeposition[:-1]].leaves()) == 1:
                _leaf_treeposition.pop(-1)
            __ = nltkptree[_leaf_treeposition]

        if len(connidx) >1: 
            token_nums = connidx.copy()
            previous = token_nums.pop(-1)
            while len(token_nums) >0:
                current = token_nums.pop(-1)
                try:
                    assert previous - current ==1
                    previous = current
                except:
                    # case 3: multiple word DC, disjoint/parallel 
                    # TBD
                    break

            # case 2: multi-word DC, continuous 
            _subtreehead =  list(nltkptree.treeposition_spanning_leaves(min(connidx), max(connidx)+1)) 
            # +1 as treeposition_spanning_leaves end is not included in search. 
            # _subtreehead.pop(-1)# pop the last element in order to get the treeposition for the head of the leaf node
            if len(nltkptree[_subtreehead]) > len(connidx):
                pass
            else: 
                while len(nltkptree[_subtreehead[:-1]].leaves()) == len(connidx) and len(_subtreehead) > 1:
                    _subtreehead.pop(-1)
            __ = nltkptree[_subtreehead]
        
        _label = __.label()
        _idx = __.treeposition()

        return (_label, _idx)

    
    def PRNT(*args, **kwargs): #input: selfcatidx = selfcatidx, nltkptree = nltkptree):
        """
        Returns the syntactic category of the parent of selfcat above. 
        
        input |selfcatidx: list - output from getfeat_selfcat, nltkptree: nltk.ParentTree
        object, features: dict - dictionary with feature values and associated label
        output | _label: int - the mapped label for the feature value, _idx: tuple - the treeposition indices
        """

        selfcatidx, nltkptree = itemgetter('selfcatidx', 'nltkptree')(args[0])

        __ = nltkptree[selfcatidx].parent()

        # in a well-formed constparse, there shouldn't be a case of there being no parentcat.     
        _label = __.label()
        _idx = __.treeposition() 
        
        return (_label, _idx)

    def LEFTSib(*args, **kwargs): #input: selfcatidx = selfcatidx, nltkptree = nltkptree, skip_punct = False):
        """
        Similar to getfeat_parentcat; returns the syntactic category of the left
        sibling  of selfcat above. 
        
        input | selfcatidx: list - output from getfeat_selfcat, nltkptree: nltk.ParentTree object, features: 
        dict - dictionary with feature values and associated label 
        output | _label: int - the mapped label for the feature value, _idx: tuple - the treeposition indices. 
        """

        selfcatidx, nltkptree = itemgetter('selfcatidx', 'nltkptree')(args[0])

        __ = nltkptree[selfcatidx].left_sibling()

        try: 
            _label = __.label()
            _idx = __.treeposition() 
        except AttributeError:
            _label='NoLeftSib'# we pad with 'NoLeftSib' to distinguish with cases where other features are not found
            _idx = None 
        
        return (_label, _idx )

    def RGHTSib(*args, **kwargs): #input: selfcatidx = selfcatidx, nltkptree = nltkptree, skip_punct = False):
        """
        Similar to getfeat_parentcat; returns the syntactic category of the right sibling  of selfcat above. 

        input | selfcatidx: list - output from getfeat_selfcat, nltkptree: nltk.ParentTree object, features: 
        dict - dictionary with feature values and associated label.
        output | _label: int - the mapped label for the feature value, _idx: tuple - the treeposition indices.
        """

        selfcatidx, nltkptree = itemgetter('selfcatidx', 'nltkptree')(args[0])

        __ = nltkptree[selfcatidx].right_sibling()

        try: 
            _label = __.label()
            _idx = __.treeposition() 
        except AttributeError:
            _label='NoRightSib'  # input: same comment as NoLeftSib above
            _idx = None 
        
        return (_label, _idx, )


##### from Lin et al 2010                                               #####

class Lin_etal: 

    def CPOS(*args, **kwargs): # input: connidx = connidx, nltkptree = nltkptree):
        """
        Returns the part of speech for the connective (corresponds to the C POS feature in Lin et al). 
        This differs from the SELF feature in P&N - e.g. for the parse (S (CC But) (NP (NNS analysts)) 
        (VP (VBP reckon)..., 
        the SELF category for 'analyst' would be NP, whereas the CPOS for 'analysts' would be NNS. 
        """

        connidx, nltkptree = itemgetter('connidx', 'nltkptree')(args[0])

        if len(connidx) == 1: 
            _idx = list(nltkptree.leaf_treeposition(connidx[0]))
            # pop last index to get to index of parent (i.e. POS tag)
            _idx.pop(-1)
            _label = nltkptree[_idx].label()
        if len(connidx) > 1: 
            # treating MWE DCs as a single unit, using SelfCat (the syntactic category of the constituent that 
            # covers all and only the words in the MWE DC) to capture this signal
            _selfcat = PitlerNenkova_Syn.SELF(args[0]) # adding args[0] here, what it's doing is passing the *args 
            # (that was passed into this getfeat_CPOS in the b1_featuresbuilder.py module) into this further call of 
            # another function.  
            _label = _selfcat[0]
            _idx = _selfcat[1]

        return (_label, _idx)


    def Prev_C(*args, **kwargs): #input: prev_nltkptree_idx = prev_nltkptree_idx, 
        # connstring = connstring, nltkptree = nltkptree):
        """
        Returns the string form of the word before the connective. (corresponds to the prev1 + C string feature in Lin et al).
         For MWE connectives, this is the string immediately to the left of the first word of the connective. 
        """
        prev_nltkptree_idx, connstring, nltkptree = itemgetter('prev_nltkptree_idx','connstring', 'nltkptree')(args[0])

        if isinstance(prev_nltkptree_idx, str):
            _idx = "No_idx"  
            prevstring = "NoPrevWord"
        else: 
            _idx = list(prev_nltkptree_idx)
            prevstring = nltkptree[_idx]
        
        _label = prevstring+ '^' +connstring
        
        return (_label, _idx)


    def PrevPOS(*args, **kwargs): # input: prev_nltkptree_idx = prev_nltkptree_idx,
        # nltkptree = nltkptree, skip_punct = False): 
        """
        Returns the part of speech for the word immediately before the connective. (corresponds to the prev1 POS feature in 
        Lin et al) For MWE connectives, this is the string immediately to the left of the first word of the connective. 
        """
        prev_nltkptree_idx, nltkptree = itemgetter('prev_nltkptree_idx', 'nltkptree')(args[0])

        if isinstance(prev_nltkptree_idx, str):  
            _idx = "No_idx"
            _label = "NoPrevPOS"
        else:     
            _idx = list(prev_nltkptree_idx)
            # pop last index to get to index of parent (i.e. POS tag)
            _idx.pop(-1)
            _label = nltkptree[_idx].label()

        return (_label , _idx)


    def PrevPOS_CPOS(*args, **kwargs): # input: prevPOS=None, CPOS=None, 
        #connidx = connidx, prev_nltkptree_idx=None, nltkptree = nltkptree, skip_punct = False): 
        """
        Returns a string that is a concatenation of the parts of speech for the word before the connective 
        and for the connective itself. (corresponds to the prev1 POS + C POS feature in Lin et al) 
        """
        
        # the inputs to *args are nested in a tuple. args[0] extracts the inputs.in this case it is initvals_dict, 
        # which we want to pass into the 2 previously written functions  PrevPOS and CPOS
        prevPOS = Lin_etal.PrevPOS(args[0])[0]
        CPOS = Lin_etal.CPOS(args[0])[0]
        _label = prevPOS+ '^' +CPOS

        return  (_label, "No_idx")


    def C_Next(*args, **kwargs): # input: next_nltkptree_idx=next_nltkptree_idx, connstring=connstring, 
        # nltkptree=nltkptree, skip_punct = False):
        """
        Returns the string form of the word immediately after the connective.(corresponds to the C string + next1  
        feature in Lin et al) For MWE connectives, this is the string immediately to the right of the last word of 
        the connective. 
        """
        next_nltkptree_idx, connstring, nltkptree = itemgetter('next_nltkptree_idx','connstring', 'nltkptree')(args[0])
        
        if isinstance(next_nltkptree_idx, str):  
            _idx = "No_idx"
            nextstring = "NoNextWord"
        else: 
            _idx = list(next_nltkptree_idx)
            nextstring = nltkptree[_idx]

        _label = nextstring+ '^' +connstring
        
        return (_label, _idx)


    def NextPOS(*args, **kwargs): # input: next_nltkptree_idx=next_nltkptree_idx, nltkptree=nltkptree, skip_punct = False):
        """
        Returns the part of speech for the word immediately after the connective. (corresponds to the next1 POS 
        feature in Lin et al)  
        """
        next_nltkptree_idx, nltkptree = itemgetter('next_nltkptree_idx', 'nltkptree')(args[0])

        if isinstance(next_nltkptree_idx, str):  
            _label = "NoNextPOS"
        else:     
            _idx = list(next_nltkptree_idx)
            # pop last index to get to index of parent (i.e. POS tag)
            _idx.pop(-1)
            _label = nltkptree[_idx].label()

        return (_label, _idx)

    def CPOS_NextPOS(*args, **kwargs): #input:  connidx = connidx, nltkptree = nltkptree, skip_punct = False): 
        """
        Returns a string that is a concatenation of the parts of speech for the connective itself and the word 
        immediately after the connective.  (corresponds to the C POS + next1 POS feature in Lin et al) 
        """

        nextPOS = Lin_etal.NextPOS(args[0])[0]
        CPOS = Lin_etal.CPOS(args[0])[0]
        _label = CPOS+ '^' +nextPOS

        return  (_label, "No_idx")


    def FullPath(*args, **kwargs): # input: connidx = connidx, nltkptree = nltkptree):
        """
        Returns the string concatenating the labels of all the nodes between the connective's parent and the 
        root of the constituency parse tree. (corresponds to the full path of C’s parent → root feature in Lin et al)
        """

        connidx, nltkptree = itemgetter('connidx', 'nltkptree')(args[0])

        fullpath = list()
        _idx = list(nltkptree.leaf_treeposition(connidx[0]))
        while len(_idx) > 1: # the CoNLL 2015 data has an additional " " root, we discard it.  
            _idx.pop(-1)
            fullpath.append(nltkptree[_idx].label())
        
        _label = "-".join(fullpath)

        return  (_label, "No_idx")


    def CompPath(*args, **kwargs): # input: connidx = connidx, nltkptree = None):
        """
        The same as FullPath, except that sequentially repetition (i.e. more than one) of a label is reduced 
        to a single occurence on the string. Returns the string concatenating, in a compressed manner, the labels 
        of all the nodes between the connective's parent and the root of the constituency parse tree. (corresponds 
        to the compressed path of C’s parent → root feature in Lin et al)
        """

        fullpath = Lin_etal.FullPath(args[0])[0]
        _fullpath_tokens = fullpath.split("-")
        
        compressedpath = []
        _current = None 
        while len(_fullpath_tokens) > 0: 
            _visiting = _fullpath_tokens.pop(0)
            if _visiting != _current:
                compressedpath.append(_visiting)
                _current = _visiting
            else: 
                continue

        _label = "-".join(compressedpath)

        return  (_label, "No_idx")


##### Li et al 2016                                                #####

class Li_etal16:
    """
    The set of features used by Li et al in their submission for the CoNLL 2016 Shared Task. 
    """
    SELF = PitlerNenkova_Syn.SELF
    Prev_C = Lin_etal.Prev_C
    PrevPOS = Lin_etal.PrevPOS
    PrevPOS_CPOS = Lin_etal.PrevPOS_CPOS
    C_Next = Lin_etal.C_Next
    NextPOS = Lin_etal.NextPOS
    CPOS_NextPOS = Lin_etal.CPOS_NextPOS


########## Helper funcs to get indices, navigate between collections ##########
########## of  Parse-class and Relation-class objects                ##########

def _CStr(Relation_obj, mapped = True):
    """
    DUPLICATE of CStr in this module. Intended for use in _make_initvars. Returns the associated number in the 
    feature-class index (features). 
    input | Relation_obj: Relation-class object - object containing information about a single discourse relation. 
    output | str - the wordform(s) in the discourse connective (MappedText or RawText)
    """
    _featname = 'CStr'
   
    if mapped == True: 
        _label = Relation_obj.Connective["MappedText"]
    if mapped == False: 
        _label = Relation_obj.Connective["RawText"] 

    return (_label, "No_idx", _featname)


def _get_connidx(Relation_obj):
    """
    Takes a Relation-class object, extracts the information about its connective
    
    input | an instance of a Relation-class 
    output | connidx: list - containing the index number (zero-indexed) that the connective's word(s) have 
    within the sentence. e.g. when handling "when and if" in "When and if Jim eats, the table becomes messier.", 
    will return [0,1,2]. 
    """
    conninfo = Relation_obj.Connective['TokenList']
    connidx = [i[-1] for i in conninfo]
    return connidx


def _get_prevnextidx_nltkptree(connidx, nltkptree):
    """
    Takes a Relation-class object, extracts the information about its connective's previous word. Used in the 
    functions to build the Lin et al set of features.

    """
    previdx = min(connidx) - 1  # if the conn is already at the start of the 
                                # sentence, then this value will be -1. the 
                                # next step to get the nltkptree index will 
                                # fail and we'd set the index to a special 
                                # string. 

    nextidx = max(connidx) + 1  # same as above, but + 1 instead

    try: prev_nltkptree_idx = nltkptree.leaf_treeposition(previdx)
    except: prev_nltkptree_idx = "NoPrev"
    try: next_nltkptree_idx = nltkptree.leaf_treeposition(nextidx)
    except: next_nltkptree_idx = "NoNext"

    return prev_nltkptree_idx, next_nltkptree_idx 


def _get_selfcatidx_nltkptree(connidx, nltkptree):
    """
    get the syntactic category of the discourse connective. For a single word connective, this is the same 
    as its part of speech tag, for MWE connectives,it is the category of the node that covers all, yet only, 
    the tokens of the MWE connective.
    """
    if len(connidx) == 1: 
        # case 1: single word DC    
        # pop the last element in order to get the treeposition for the head of the leaf node
        _leaf_treeposition = list(nltkptree.leaf_treeposition(connidx[0]))
        _leaf_treeposition.pop(-1)
        __ = nltkptree[_leaf_treeposition]

    elif len(connidx) >1: 
        # case 2: multi-word DC, continuous 
        _subtreehead = nltkptree.treeposition_spanning_leaves(min(connidx), max(connidx)+1)
        __ = nltkptree[_subtreehead]

    else:
        # case 3: multiple word DC, disjoint/parallel 
        # TBD 
        pass

    selfcatidx = __.treeposition()
    
    return selfcatidx


def _retrieve_parsestring(Relation_obj, Parse_dict, gold):
    """
    Return the constituency parse of the sentence (in PTB-style s-tree format)
    """
    DocID, SentID = Relation_obj.DocID, Relation_obj.SentID
    if gold == True: synparse = Parse_dict[DocID][SentID].ConstTree_Gold
    if gold == False: synparse = Parse_dict[DocID][SentID].ConstTree_Auto

    return synparse


def _make_nltkptree(synparse):
    """
    Given the syntactic parse of a sentence, in a single-line string form, return the NLTK ParentedTree object. 
    """
    nltkptree = tree.ParentedTree.fromstring(synparse)

    def _delete_tracenodes(nltkptree):
        """
        Helper function to remove trace nodes from gold constituency parses. This is used in PitlerNenkova_Syn.SELF 
        inspired by but replaced search for * in leaf with search for -NONE- in parent, tested before deploying 
        here: https://stackoverflow.com/questions/33939486/how-to-identify-and-remove-trace-trees-from-nltk-trees
        """
        for sub in reversed(list(nltkptree.subtrees())): 
            if sub.height() == 2 and sub.label().startswith("-NONE-"):  
                parent = sub.parent()
                while parent and len(parent) == 1:
                    sub = parent
                    parent = sub.parent()
                del nltkptree[sub.treeposition()]

    _delete_tracenodes(nltkptree)
    return nltkptree


def _make_initvars(Relation_obj, Parse_dict, gold): 
    """
    Generate the variables and values necessary to initialise all the functions in this module.  
    """
    connstring = _CStr(Relation_obj)[0]
    connidx = _get_connidx(Relation_obj)
    synparse = _retrieve_parsestring(Relation_obj, Parse_dict, gold)
    nltkptree = _make_nltkptree(synparse)
    prev_nltkptree_idx, next_nltkptree_idx = _get_prevnextidx_nltkptree(connidx, nltkptree)
    selfcatidx = _get_selfcatidx_nltkptree(connidx, nltkptree)
    try: 
        assert len(Parse_dict[Relation_obj.DocID][Relation_obj.SentID].Words) == len(nltkptree.leaves()) 
    except AssertionError as e:
        e.args += ('There is a misalignment between the constituency parse and the tokenisation of the raw text. This could to the removal of traces in this constituency parse (if using gold-label PTB parses during the experiment/model training stage). The docid is {} and the sentid is {}.'.format(Relation_obj.DocID, Relation_obj.SentID),)
        raise
    
    return connstring, connidx, prev_nltkptree_idx, next_nltkptree_idx, synparse, nltkptree, selfcatidx


if __name__ == "__main__":
    
    pass
