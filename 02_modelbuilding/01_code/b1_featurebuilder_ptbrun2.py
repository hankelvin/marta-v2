# -*- coding: utf-8 -*-
import b1_featurebuilder_ptbrun1
from nltk import tree 
from operator import itemgetter 


"""
##### Notes on use                                                         #####
1. This module relies on the nltk.tree package https://www.nltk.org/_modules/nltk/tree.html. for handling PTB-style constituency parsing. 
2. This module utilises two different index formats for obtaining the feature values. 
 a. The first is the WordToken indices used in the CoNLL 2015 Shared Task. For each word, this is a fixed-sized (5-element) tuple. 
 b. The second is the treeposition indices found in the NLTK.tree.ParentTree class. It is a list of variable length providing the position of the word within the tree for the constituency parse tree. 
3. The naming convention for the functions the feature classes corresponds to the feature names (e.g. part of speech for connective from Lin et al 2010, CPOS), so they can be easily extracted from a module inspection in b_featurebuilder.py. 

"""

######### Functions to build features in connective identification ##########
##### Constituency parse-based features                                 #####
##### from Pitler & Nenkova 2009                                        #####

class PitlerNenkova_Conn: 
    CStr = b1_featurebuilder_ptbrun1.PitlerNenkova_Conn.CStr

# class PitlerNenkova_Syn:

    # the following features present in Run 1 are removed in Run 2. This is because there is no direct equivalent in the UD framework. 
    # SELF = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.SELF
    # PRNT = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.PRNT 
    # LEFTSib = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.LEFTSib
    # RGHTSib = b1_featurebuilder_ptbrun1.PitlerNenkova_Syn.RGHTSib


##### from Lin et al 2010                                               #####

class Lin_etal: 

    def CPOS(*args, **kwargs): #= b1_featurebuilder_ptbrun1.Lin_etal.CPOS
        """
        Returns the part of speech of the connective candidate.
        """
        connidx, nltkptree = itemgetter('connidx', 'nltkptree')(args[0])
        if len(connidx) == 1: 
            _idx = list(nltkptree.leaf_treeposition(connidx[0]))
            # pop last index to get to index of parent (i.e. POS tag)
            _idx.pop(-1)
            _label = nltkptree[_idx].label()
        else: # i.e. MWE DC
            
            pos_idxset = list()
            [pos_idxset.append(list(nltkptree.leaf_treeposition(i))) for i in connidx] # add the leaf idx for each node in the conn
            [i.pop() for i in pos_idxset] # pop to get the syn cat of each leaf
            label = set([nltkptree[i].label() for i in pos_idxset]) #placing in a set to manage sparsity
            _idx = pos_idxset
            _label = "_".join(label)

        return (_label, _idx) 

    Prev_C = b1_featurebuilder_ptbrun1.Lin_etal.Prev_C
    PrevPOS = b1_featurebuilder_ptbrun1.Lin_etal.PrevPOS
    PrevPOS_CPOS = b1_featurebuilder_ptbrun1.Lin_etal.PrevPOS_CPOS
    C_Next = b1_featurebuilder_ptbrun1.Lin_etal.C_Next
    NextPOS = b1_featurebuilder_ptbrun1.Lin_etal.NextPOS
    CPOS_NextPOS = b1_featurebuilder_ptbrun1.Lin_etal.CPOS_NextPOS

    # the following features present in Run 1 are removed in Run 2. This is because there is no direct equivalent in the UD framework. 
    # FullPath = b1_featurebuilder_ptbrun1.Lin_etal.FullPath 
    # CompPath = b1_featurebuilder_ptbrun1.Lin_etal.CompPath

##### Li et al 2016                                                #####

class Li_etal16:
    """
    The set of features used by Li et al in their submission for the CoNLL 2016 Shared Task. 
    """
    Prev_C = Lin_etal.Prev_C
    PrevPOS = Lin_etal.PrevPOS
    PrevPOS_CPOS = Lin_etal.PrevPOS_CPOS
    C_Next = Lin_etal.C_Next
    NextPOS = Lin_etal.NextPOS
    CPOS_NextPOS = Lin_etal.CPOS_NextPOS
    # the following features present in Run 1 are removed in Run 2. This is because there is no direct equivalent in the UD framework. 
    # SELF = PitlerNenkova_Syn.SELF

########## Helper funcs to get indices, navigate between collections ##########
########## of  Parse-class and Relation-class objects                ##########

_CStr = b1_featurebuilder_ptbrun1._CStr
_get_connidx = b1_featurebuilder_ptbrun1._get_connidx


_get_prevnextidx_nltkptree =b1_featurebuilder_ptbrun1._get_prevnextidx_nltkptree
_get_selfcatidx_nltkptree = b1_featurebuilder_ptbrun1._get_selfcatidx_nltkptree
_retrieve_parsestring = b1_featurebuilder_ptbrun1._retrieve_parsestring
_make_nltkptree = b1_featurebuilder_ptbrun1._make_nltkptree
_make_initvars = b1_featurebuilder_ptbrun1._make_initvars


if __name__ == "__main__":
    # script tests below
    import dill 
    with open('03_data/en/explicit_connectives/ExpConn_posexp_dev.dill', 'rb') as f:
        Relation_dict = dill.load(f)
        Relation_obj = Relation_dict[0]
    with open('03_data/en/pdtb_conll_data/ParsePDTB_dict_dev.dill', 'rb') as f:
        Parse_dict = dill.load(f)

    initvars = ['connstring', 'connidx', 'prev_nltkptree_idx', 'next_nltkptree_idx', 'synparse', 'nltkptree', 'selfcatidx' ]
    initvals = _make_initvars(Relation_obj, Parse_dict, gold=True)

    initvars_dict = {k:v for k,v in zip(initvars, initvals)}
    print('CStr', PitlerNenkova_Conn.CStr(initvars_dict))
    print('CPOS', Lin_etal.CPOS(initvars_dict))
    print('C_Next', Lin_etal.C_Next(initvars_dict))
    print('CPOS_NextPOS', Lin_etal.CPOS_NextPOS(initvars_dict))
    print('NextPOS', Lin_etal.NextPOS(initvars_dict))
    print('Prev_C', Lin_etal.Prev_C(initvars_dict))
    print('PrevPOS', Lin_etal.PrevPOS(initvars_dict))
    print('PrevPOS_CPOS', Lin_etal.PrevPOS_CPOS(initvars_dict))
    pass
