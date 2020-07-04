# -*- coding: utf-8 -*-
#! /usr/bin/env python3
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from inspect import getmembers, isfunction, isclass
from pprint import pprint

"""
########## Notes on usage                                           ##########
1. The resulting scipy npz and numpy npy files for every featureset contains the predictors (X) and the labels (y). The rows in the npz and npy files have not be shuffled, and begin with the positive examples before the negative examples start. Use a shuffle method (after concatenating feature sets) in the next step of the workflow (before applying a machine learning algorithm).
2. One hot encoding:
    a. is done as a standard step for the dataset. while logistic regression takes categorical values that are not-binary, other typically used non-linear classifiers don't. 
    b. is done with drop_first = True, to avoid multicolinearity issues
    c. is done with sparse = False. the OHE outputs are being put through pandas for easier handling and the sparse pandas take very long to be handled, so we return it into numpy form before storing. 
3. For interaction features: 
    a. Only interaction features are produced (i.e. products between input features), and up to the interaction degree (we retain the default setting which is at degree 2, modeling joint probability of two features)
    b. a bias term is included (include_bias = True; a columns of 1s is included in the generated featureset). Some of the experiments are using generalised linear models (MaxEnt/LogReg) so this allows the model to tune a intercept term.  
    b. the "input" singleton features captured in the interaction datasets. This means that the interaction featureset can be used independently without having to stack the input featureset(s) used to generate the interaction features. 
4. IMPORTANT: scroll down to if __name__ == "__main__": section and set the     
    environment variables before running this module to produce the datasets.
"""


######### Functions to produce featuresets                          #########

def make_featureset(Relation_objs, Parse_dict, gold, feature_funcs, featurebuilder, framework = 'PTB', *args, **kwargs):
    """
    Given collections of (1) RelationPDTB and (2) ParsePDTB objects, as well as specifications on what features to produce, return a pandas dataframe with a raw set of features (i.e. no one-hot-encoding, no interaction features).
    Parameters:
    - Relation_objs (list): 
    - Parse_dict (dict): a dictionary containing ParsePDTB objects. The first level keys are document IDs, the second level are sentence IDs within each document.  
    - gold (bool): True if gold-label syntactic information available. 
    - feature_funcs (list): a collection of function objects used for producing the featureset.
    - framework (str): the syntactic framework being used.
    Return: 
    df_raw (pandas DataFrame): containing the features produced for each sample.
    """
    rawfeatureset, featname_list = _get_rawfeatureset(Relation_objs, Parse_dict, gold, feature_funcs, featurebuilder, framework=framework)
    # 1. place rawfeaturesets into a pandas DF
    df_raw = pd.DataFrame(rawfeatureset, columns = featname_list)
    return df_raw

def do_onehotencoding(df_raw):
    """
    Takes raw features and generates the equivalent one-hot representation. Importantly, stores the resulting featurenames for (1) matching in order to test model, or use in production; and/or (2) error analysis. 
    Parameters:
    - df_raw (pd.DataFrame): the output of make_featureset.
    Returns: 
    - df_OHE (pd.DataFrame): the one-hot encoded data.
    - df_labelsdict (dict): containing the featurenames after the one-hot encoding.  
    """
    # 2. one-hot-encode and dropping first OHE variable for every feature
    df_OHE = pd.get_dummies(df_raw, sparse=False, drop_first=True) 
    df_labelsdict = {k:v for k,v in enumerate(df_OHE.columns)}
    return df_OHE, df_labelsdict

def make_interactionfeatures(stackedfeats, featnames, interaction_degree = 2, interaction_only=True):
    """
    Produces the interaction features that were used by Pitler & Nenkova (2009) in their featuresets.
    """
    # 3. generate interaction features (used in exps for Pitler & Nenkova featureset) 
    pf = PolynomialFeatures(degree=interaction_degree, interaction_only=interaction_only)

    interacted = pf.fit_transform(stackedfeats)
    interacted_labels = pf.get_feature_names(input_features = featnames)

    return interacted, interacted_labels 

def _get_rawfeatureset(Relation_objs, Parse_dict, gold, feature_funcs, featurebuilder, framework='PTB', *args, **kwargs):
    """
    helper function that allows loading multiple feature_builderPTB/UD.py funcs in order to build the featureset. 

    no interaction features if interact_deg = 0. By default here, the interaction_only setting in sklearn's PolynomialFeatures is set to True - i.e. original features are not produced in the return output. This is coherent with approaches using generalised linear model classifiers (i.e. not introducing collinearity). 
    """
    print('Syntactic framework in use:', framework)
    rawfeaturesets = list()
    featname_list = [i[0] for i in feature_funcs]
    for eachRelationobj in Relation_objs:  
        # obtain the initial variables 
        if framework == 'PTB':
            initvars = ['connstring', 'connidx', 'prev_nltkptree_idx', 'next_nltkptree_idx', 'synparse', 'nltkptree', 'selfcatidx']
        if framework in ['UD1','UD2']:
            initvars = ['connstring', 'connidx', 'deptree', 'wordslist', 'sentlength', 'gold_']
        
        initvals = featurebuilder._make_initvars(eachRelationobj, Parse_dict, gold)
        initvars_dict = {k:v for k,v in zip(initvars, initvals)}

        feats_row = list()
        # 1. run all funcs  specified. each _feat_val output from the func is appended to the feats_row list.
        for eachfeature_func in feature_funcs:
            func = eachfeature_func[1](initvars_dict)
            _feat_val = func[0]
            feats_row.append(_feat_val)
        # 2. full feats_row is appended to the master rawfeaturesets list. 
        rawfeaturesets.append(feats_row)
    
    # convert rawfeaturesets to np array for easier handling 
    rawfeaturesets = np.array(rawfeaturesets)

    return rawfeaturesets, featname_list 


if __name__ == "__main__":
    # steps to produce the datasets for the experiment phase.
    import dill, json 
    from os import makedirs, path
    from scipy import sparse
    import b1_featurebuilder_ptbrun1, b1_featurebuilder_ptbrun2, b2_featurebuilder_udrun2

    FRAMEWORK = 'UD1'
    GOLD = False # whether to use gold constituency/dependency parses or auto generated constituency/dependency parses. 
    RUNNUM = '2'
    if GOLD == True: EXP_NAME = '{}_{}_Run{}'.format(FRAMEWORK,'Gold', RUNNUM)
    elif GOLD == False:  EXP_NAME = '{}_{}_Run{}'.format(FRAMEWORK, 'Auto', RUNNUM)
    else: raise ValueError ('Check setting for GOLD variable, it has to be either True or False.') # ensure that value for GOLD matches value for EXP_NAME
    featuremodules = {  'PTB_Gold_Run1': b1_featurebuilder_ptbrun1 ,
                        'PTB_Auto_Run1': b1_featurebuilder_ptbrun1,

                        'PTB_Gold_Run2': b1_featurebuilder_ptbrun2,
                        'PTB_Auto_Run2': b1_featurebuilder_ptbrun2,  
                        'UD1_Gold_Run2': b2_featurebuilder_udrun2, 
                        'UD1_Auto_Run2': b2_featurebuilder_udrun2, 
                        
                        'UD1_Gold_Run3': b2_featurebuilder_udrun3, 
                        'UD1_Auto_Run3': b2_featurebuilder_udrun3, }
    FEATUREBUILDER = featuremodules[EXP_NAME] # setting the module that will be used in 0b below. 

    # 0b. load the sets of functions for producing the features
    # for all b1_featurebuilder_ptb functions
    FEATURE_FUNCS = {class_[0]: [func for func in getmembers(class_[1], isfunction)] for class_ in getmembers(FEATUREBUILDER, isclass) if class_[0] != 'itemgetter'} #itemgetter was being imported in FEATUREBUILDERs :P
    pprint(FEATURE_FUNCS)


    # # for just one featureclass (in this case, the Li et al 2016 functions)
    # FEATURE_FUNCS = {class_[0]: [func for func in getmembers(class_[1], isfunction)] for class_ in getmembers(FEATUREBUILDER, isclass) if class_[0] == 'Li_etal16'}
    # pprint(FEATURE_FUNCS)

    def make_traintestsets(datasets=['train', 'dev', 'test'], feature_funcs = FEATURE_FUNCS, featurebuilder = FEATUREBUILDER, lang="zz", gold = GOLD, exp_name = EXP_NAME, framework = FRAMEWORK): 
        """
        helper function to run make_featureset and OHE across train and test  
        """
        # 0a. load the set of Relation-class objects and the associated dictionary of Parse-class objects. 
        for dataset in datasets:
            with open("./03_data/{}/explicit_connectives/ExpConn_posexp_{}.dill".format(lang, dataset), 'rb') as f:
                    conns_posexamples = dill.load(f)
            with open("./03_data/{}/explicit_connectives/ExpConn_negexp_{}.dill".format(lang, dataset), 'rb') as f:
                    conns_negexamples = dill.load(f)
            with open("./03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill".format(lang, dataset), 'rb') as f:
                    ParsePDTB_dict = dill.load(f)

            # 1. Basic set of features in Pitler & Nenkova 2009, Lin et al 2010, Li et al 2016 (i.e. without interaction features)
            for featureclass in feature_funcs: 
                print("I am processing this set: {} and this featureclass: {}".format(dataset, featureclass))

                # a. positive examples 
                featureset_pos  = make_featureset(conns_posexamples, ParsePDTB_dict, gold, FEATURE_FUNCS[featureclass], featurebuilder = featurebuilder, framework = FRAMEWORK) 
                y_label_pos = np.zeros(featureset_pos.shape[0]) + 1 # generate y_labels 
                print("# pos:", len(y_label_pos))

                # b. negative examples 
                featureset_neg = make_featureset(conns_negexamples, ParsePDTB_dict, gold, FEATURE_FUNCS[featureclass], featurebuilder = FEATUREBUILDER,framework = FRAMEWORK) 
                y_label_neg = np.zeros(featureset_neg.shape[0])    # generate y_labels 
                print("# neg:", len(y_label_neg))

                # c. merge positive and negative featuresets and labels  
                featureset_all = pd.concat([featureset_pos, featureset_neg])
                featureset_all.reset_index(inplace=True, drop=True) # resetting index to remove duplicate indices from the merging 
                y_labels = np.concatenate([y_label_pos, y_label_neg])

                # d. do one hot encoding 
                X_featureset, featlabels = do_onehotencoding(featureset_all)
                __X_featureset = X_featureset.to_numpy()

                #############################################
                ######### Start data_split treatment#########

                if dataset == datasets[0]: 
                    # the next assertion is a safety check for duplicate feature names. however, since the features were properly named before OHE, we don't expect repetitions. 
                    try: 
                        assert sorted(featlabels.values()) == sorted(set(featlabels.values()))
                        # recall that the featlabels result of do_onehotencoding is a dict with numbered keys, the values are the feature names. the intent is to have a stable numbered order of the features which can be exported via json 
                    except AssertionError as e:
                        e.args += "There are duplicate feature names in the {}dataset after one-hot-encoding, this will lead to a data integrity issue in the construction of the next (test)set.".format(dataset)
                        raise 
                    # save the featurenames to a global variable
                    globals()['featlabels_'+datasets[0]+'_'+featureclass] = featlabels

                if dataset != datasets[0]:
                    # the next assertion is a safety check for duplicate feature names. however, since the features were properly named before OHE, we don't expect repetitions.  
                    try: 
                        assert sorted(featlabels.values()) == sorted(set(featlabels.values()))
                        # different from the featlabels_train assertion above
                    except AssertionError as e:
                        e.args += "There are duplicate feature names in the {}dataset after one-hot-encoding, this will lead to a data integrity issue in the construction of the current (test)set.".format(dataset)
                        raise

                    # only handle feature names that are in the train list, effectively dropping those not in the train list
                    num_rows    = __X_featureset.shape[0]
                    trainsetlabels = globals()['featlabels_'+datasets[0]+'_'+featureclass] # recall it is stored as a dict with numbered keys, the values are the feature names. use .values() to access the featurenames
                    num_columns = len(trainsetlabels)
                    print('rows, columns:', num_rows, num_columns)
                    np_empty = np.zeros([num_rows, num_columns], dtype=int)

                    # 
                    trainsetlabels_list = list(trainsetlabels.values())
                    # trainsetlabels_idx2word = trainsetlabels
                    trainsetlabels_word2idx = {trainsetlabels[i]: i for i in trainsetlabels}
                    print('length of trainsetlabels_word2idx', len(trainsetlabels_word2idx))

                    totreat_word2idx = {featlabels[i]: i for i in featlabels if featlabels[i] in trainsetlabels_list}
                    print('length of totreat_word2idx', len(totreat_word2idx))

                    for featlabel in totreat_word2idx:
                        __X_featureset_colidx = totreat_word2idx[featlabel]
                        col_source = __X_featureset[:,__X_featureset_colidx]

                        trainset_colidx = trainsetlabels_word2idx[featlabel]
                        np_empty[:,trainset_colidx] = col_source

                    print('CHECK1a: df_empty: rows, columns:', np_empty.shape)
                    print('CHECK1b: trainsetlabels: #columns:', len(trainsetlabels))

                    X_featureset = None
                    X_featureset = np_empty.copy()
                    print('sum of X_featureset', sum([sum(i) for i in X_featureset]))

                    print(X_featureset)
                    featlabels = trainsetlabels
                    print('CHECK2: X_featureset: rows, columns:', X_featureset.shape)
                
                ######### End data_split treatment  #########
                #############################################

                # e. write to folder and file 
                print("Writing to files")
                dir_name = './02_modelbuilding/02_output/{}/{}/'.format(lang,exp_name)+featureclass
                if not path.exists(dir_name):
                    makedirs(dir_name)
                
                # convert to scipy csr and save as scipy npz file
                X_featureset = sparse.csr_matrix(X_featureset)
                sparse.save_npz(dir_name + "/{}_{}".format('X', dataset),  X_featureset)
                
                np.save(dir_name + "/{}_{}".format('y', dataset),  y_labels, allow_pickle=False)

                print(featureclass, dataset, )
                print('numpy exported files')
                # check that npz file written has the same shape as X_featureset
                saved_X = sparse.load_npz(dir_name + "/{}_{}.npz".format('X', dataset))
                print(saved_X.shape, X_featureset.shape)
                assert saved_X.shape == X_featureset.shape
                print('CHECK3: shape of scipy.sparse file saved is the same as X_featureset')

                with open(dir_name + "/README.md", 'w+') as f:
                    [f.write(i[0]+"\n") for i in FEATURE_FUNCS[featureclass]]
                    print('featureclass file written')
                
                with open(dir_name + "/{}_featnames.json".format(dataset), 'w+') as f:
                    json.dump(featlabels, f)
                    print('featurenames file written')

                pprint(X_featureset)
    
    # run make_traintestsets function 
    make_traintestsets(datasets=['train', 'dev', 'test'], lang = 'en', gold = GOLD)

    # 2. Interaction features in Pitler & Nenkova 2009 
    def make_traindevtestsets_interaction(new_featureclass_name, lang = 'en', datasets=['train', 'dev', 'test'], featureclasses = ['PitlerNenkova_Conn', 'PitlerNenkova_Syn'], exp_name = EXP_NAME): 
        """
        helper function to run make_interactionfeatures across train, dev and test  
        """
        for dataset in datasets:
            print('processing {} set'.format(dataset))
            to_interact = dict() 
            for featureclass in featureclasses: 
                # a. load already built features saved in scipy npz and numpy npy format
                X_filepath = './02_modelbuilding/02_output/{}/{}/{}/{}_{}.npz'.format(lang, exp_name, featureclass, 'X', dataset)
                y_filepath = './02_modelbuilding/02_output/{}/{}/{}/{}_{}.npy'.format(lang, exp_name, featureclass, 'y', dataset)

                X_featureset, y_labels = sparse.load_npz(X_filepath), np.load(y_filepath)
                X_featureset = X_featureset
                # b. load corresponding features names
                featname_filepath = './02_modelbuilding/02_output/{}/{}/{}/{}_featnames.json'
                with open(featname_filepath.format(lang, exp_name, featureclass, dataset), 'rb') as f:
                    featnames = json.load(f)

                to_interact[featureclass] = (X_featureset, y_labels, featnames)
        
            # c. stacking featuresets
            __featnames = [to_interact[i][2].values() for i in to_interact]
            featnames = list()
            for elem in __featnames: 
                featnames.extend(elem)
            stackedfeats = sparse.hstack([to_interact[i][0] for i in to_interact]) #[np.array(to_interact[i][0]) for i in to_interact][0]
            print('Dimensions before interaction:', stackedfeats.shape)

            # d. run make_interactionfeatures on joined featureset
            X_featureset_interact, featnames_interact = make_interactionfeatures(stackedfeats, featnames=featnames)
            print('Dimensions after interaction:', X_featureset_interact.shape)

            # e. write to folder and file 
            dir_name = './02_modelbuilding/02_output/{}/{}/'.format(lang,exp_name)+new_featureclass_name
            if not path.exists(dir_name):
                makedirs(dir_name)

            # convert to scipy csr and then save as scipy npz
            X_featureset_interact = sparse.csr_matrix(X_featureset)
            sparse.save_npz(dir_name + "/{}_{}".format('X', dataset),  X_featureset_interact)
     
            np.save(dir_name + "/{}_{}".format('y', dataset), y_labels, allow_pickle=True)
            with open(dir_name + "/{}_featnames.json".format(dataset), 'w+') as f:
                json.dump(featnames_interact, f)
            with open(dir_name + "/README.md", 'w+') as f:
                f.write('This set of interaction features comprise the following feature classes: \n')
                [f.write(i[0]+"\n") for i in featureclasses]
    
    # # run make_traindevtestsets_interaction 
    if RUNNUM == '1' : # interaction features not possible for Run 2 because there are no PitlerNenkova_Syn features generated there. 
        # SynSyn
        make_traindevtestsets_interaction(new_featureclass_name = 'PitlerNenkova_SynSyn', lang = 'en', datasets=['train', 'dev','test'], featureclasses = ['PitlerNenkova_Syn', 'PitlerNenkova_Syn'], exp_name = EXP_NAME)

        # ConnSyn
        make_traindevtestsets_interaction(new_featureclass_name = 'PitlerNenkova_ConnSyn', lang = 'en', datasets=['train', 'dev','test'], featureclasses = ['PitlerNenkova_Conn', 'PitlerNenkova_Syn'], exp_name = EXP_NAME)