# MarTa v2

Marker Tagger version 2, a discourse connective identification module with potential for multilingual support. This is a collection of work by Kelvin Han, Phyllicia Leavitt and Srilakshmi Balard during the first year of our masters in Natural Language Processing at the [Institut des sciences du Digital, Management & Cognition (IDMC)](http://institut-sciences-digitales.fr/), which is part of the Université de Lorraine in Nancy, France. Our literature review, experimental results and findings are described in ["Comparing PTB and UD information for PDTB discourse connective identification"](https://jep-taln2020.loria.fr/article-188/) which was accepted for the 22nd edition of RECITAL, the French student NLP conference organised in Nancy, France in 2020. The experiment and production code implementations are by Kelvin Han. 

## 1. Notes for production usage
A production-use pipeline is available [here](https://gitlab.inria.fr/andiamo/marta-v2/-/blob/master/01_production/main.py). It is suitable for use as an import or on the command line. To use it, you will need to specify a directory path containing one or more documents to be processed. Processing is seamless to the user and covers (1) sentence segmentation, (2) tokenisation, (3) syntactic parsing (both constituency parsing and dependency parsing options are provided), (4) identification of connective candidates, (5) production of featuresets, and (6) prediction using trained models. 

Two trained models are provided, [one](https://gitlab.inria.fr/andiamo/marta-v2/-/tree/master/02_modelbuilding/01_code/models/model_PTB_Auto_Run1.model) for constituency parsing input and the [other](https://gitlab.inria.fr/andiamo/marta-v2/-/tree/master/02_modelbuilding/01_code/models/model_UD1_Auto_Run2.model) for dependency parsing input. The model for constituency parses uses the featureset described by [Li et al, 2016](https://www.aclweb.org/anthology/K16-2008/), which is to our knowledge, the reported state-of-the-art for a standalone discourse connective identification module. The model for dependency parses replaces the Penn Treebank (PTB) part-of-speech (POS) tags used by Li et al, 2016 with finer-grained Universal Dependencies (UD) POS tags. One feature (SELF Category) in Li et al, 2016 -- for which there is no direct equivalent in UD -- was not included (see Section 3.1 of our paper). 
 
## 2. Notes for experiments and production model training

### a. Data
#### PDTB ConLL data

The training data comes from the 2015 and 2016 shared tasks on shallow discourse parsing:

- Description of the data: https://www.cs.brandeis.edu/~clp/conll15st/ and https://www.cs.brandeis.edu/~clp/conll16st/
- Proceedings 2015: https://www.aclweb.org/anthology/K15-2001
- Proceedings 2016: https://www.aclweb.org/anthology/K16-2001

Note that:

- There are 100 connectives in the PDTB but you'll find more when looking at the annotations because they annotated raw text (ie 'a year after' is the raw text for the connective 'after'). A baseline using only the connective as feature should use 'after' and not the raw text. A head mapper is described in the blog for the shared tasks.
- You can find multi-word connectives ('if and when') but also multiple separate connectives ('but when')
- There are a few two part/parallel connectives (namely: 'either ... or', 'if ... then', 'neither ... nor', 'on the one hand...on the other hand'). Note that these are not handled currently handled in their entirety here (only their first parts are). There are 49 instances of such parallel connectives across the entire PDTB.

Some statistics: 

- A README_<dataset>.md file containing a tally of the positive and negative examples can be found in the _/explicit_connectives_ folder for each language. For instance, these files can be found [here](https://gitlab.inria.fr/andiamo/marta-v2/-/tree/master/03_data/en/explicit_connectives) for the English data.
- In addition, we provide a [description](https://gitlab.inria.fr/andiamo/marta-v2/-/blob/master/03_data/README.md) of how we obtained approximations of gold-label UD v1.0 parses using gold-label PTB parses. 
- The following table provides an overview of the DCs found in the English PDTB datasets: 


|Dataset  | Positive  | Negative |
|---|---|---|
| train | 14,722 | 36,439 |
| dev | 680 | 1,538 |
| test | 923 | 2,065 |

#### Copyright 
- The training, development and test data - PDTB and CDTB - has to be accessed from the Linguistics Data Consortium (LDC). The FDTB is however freely available. Access to the gold PTB and CTB parses are also through LDC. Gold FTB parses can be obtained from [here](http://ftb.linguist.univ-paris-diderot.fr/index.php?langue=en) after registration. 
- Certain code snippets, such as in our c2_utils module, are adopted from the tools provided by the  CoNLL 2015 and 2016 organisers.  

### b. Experimental set-up

#### Runs 

|Run #   | Description  | Gold parses? |
|---|---|---|
| 1 |  replication of the experiments as described in (i) Pitler & Nenkova 2009, (ii) Lin 2010, and (iii) Li et al 2016. The experiments in this run are done only with constituency parses | Done with both gold and automatic parses from raw text   |
| 2 | this run involved a subset of the features used in Run 1 above. The subset is the maximal set of features where comparable information can be obtained from dependency parses. It leaves out features related to clause and phrase-level syntactic labels that can be found in constituency parses. This approach was taken so as to isolate the impact of using coarser-grained Universal Dependencies part of speech tags compared to P/F/C-TB POS tags.  | conducted on both the UD and PTB-style datasets, with gold and automatic parses for both  |

#### Tracking results of Runs 
Currently, experiments are set up to initiate from a Jupyter notebook (d1_analysis.ipynb). By changing the environment variables at the start of the notebook, you can easily move between Runs as well as the use of (i) PTB-style/UD, (ii) gold/auto parses. An [mlflow](www.mlflow.com) pipeline has been set up within the notebook to capture all parameters, metrics and models of the experiments. 

To access the parameters, metrics and models: on command line, navigate to the [01_code folder](https://gitlab.inria.fr/andiamo/marta-v2/-/tree/master/02_modelbuilding/01_code) and use the command __$ mlflow ui__. This starts a local host to serve the mlflow interface. On a webbrowser navigate to [http://localhost:5000](http://localhost:5000) and the mlflow client will load. The left hand side of the client will list the set of experiments conducted. One can scroll through these to obtain the information on the parameters and metrics of each run within an experiment. 

### c. Code, modules and dependencies 

#### Version control 
- Code is version controlled here on the GitLab repository
- Datasets are version controlled with (DVC)[www.dvc.org].

#### Modules 
| Group | Module | Function | 
|---|---|---|
| A | a_preprocessory.py |  | 
|  | a1_dataloader.py |  | 
|  | a2_parsers.py | | 
|---|---|---|
| B | b_featuresbuilder.py |  | 
|  | b_featuresbuilder_ptbrun1.py |  | 
|  | b_featuresbuilder_ptbrun2.py |  | 
|  | b_featuresbuilder_udrun2.py |  | 
|---|---|---|
| C | c2_utils.py |  | 
|  | c3a_exploratorydataanalysis-gold.ipynb | 
|  |  c3a_exploratorydataanalysis-auto.ipynb | 
|---|---|---|
| D | d_model_tuning.py |  | 
|  | d1_analysis.ipynb |  | 


#### Notes for modules 

##### Group A: Data structure, preprocessing steps 
1. A set of classes have been defined in the __a1_dataloader.py__ module; these classes are used to store information about: (i) the sentences in the P/F/C-DTB datasets (in Parse-class objects); and (ii) the positive and negative examples of explicit DCs (in Relation-class objects). Throughout the modules, we work with Parse-class and Relation-class objects in collections (in a dictionary, and a list respectively)
2. The lowercased and the mapped text form of the explicit discourse connective is being used when searching for DCs. This means that more negative examples will be identified (when compared to using the raw text of the ). The use of lowercase is justified because many of the featuresets include features (LeftSibling, PreviousPOS) that will return unique values if there is no word preceding it (i.e. the DC example is at the start of the sentence).
3. In addition, when searching for connective candidates, the internally possible candidate DCs are also returned. This is particularly pertinent for complex DCs that are comprised of more than one single/multi-word DC. For instance, 'when and if', 'if and when', where all three tokens in the larger DC are also DC candidates in their own right. This has no impact on the count of the positive examples (as annotated in the P/F/C-DTB), but results in more negative examples being generated. 
4. All of the required parses for the experiments (PTB-style as well as UD 1.0) are produced (automatic parses from raw text) or converted (from gold PTB-style parses) when a_preprocessor.py is run. 

#### Group B: Feature builders 
1. Each of the run has a separate feature-building script. These are called on by [b_featuresbuilder.py](https://gitlab.inria.fr/andiamo/marta-v2/-/blob/master/02_modelbuilding/01_code/b_featuresbuilder.py) when building the featureset for each of the different experiments.

#### Group C: Utilities 
1. [c2_utils.py](https://gitlab.inria.fr/andiamo/marta-v2/-/blob/master/02_modelbuilding/01_code/c2_utils.py) contain a set of functions to load the PDTB data in the format provided by the CoNLL 2015 and CoNLL 2016 Shared Task organisers. Some parses were missing from the data provided and scripts to produce these can be found here as well. 
2. Two sets of Jupyter (.ipynb) notebooks are provided with scripts to carry out exploratory data analysis on the prototypical properties of positive and negative DCs under the UD1.0 framework. One notebook examines the structures from using dependency parses converted from gold PTB-style parses, and the other notebook examines the structures from automatically generated parses (using the latest available version of Stanford CoreNLP suite of parsers).


## 3. Dependencies 
Python 3.6
### a. For training and experiments 
- stanfordnlp (specifically, server.CoreNLPClient)
- nltk (specifically, the ParentedTree class)
- pandas  
- numpy 
- scipy.sparse
- sklearn
- mlflow
- codecs, json, dill, copy, re, time, inspect, pprint, operator, warnings, sys, os, subprocess
- Stanford CoreNLP 3.9.2 (specifically its EnglishGrammaticalStructure and ChineseGrammaticalStructure classes)
- BerkeleyParser v1.7

### b. For production use
- spacy (including en_core_web_sm)
- [benepar](https://github.com/nikitakit/self-attentive-parser)
- tensorflow==1.15 (the version is required due to Benepar dependencies; check if TensorFlow V2 [PR](https://github.com/nikitakit/self-attentive-parser/issues/60) has since been merged.)


## Potential extensions: 
1. a script to read FDTB information in xml format and populate the Parse and Relation-class objects used here. 
2. ensure a_preprocessor.py can and process read CDTB information (and FDTB's xml info converted from xml)
3. adjust the settings for the French and Chinese parser/converters 
4. implement Malt parser for Chinese (to get Auto PTB-style parses)
# MarTa-v2
# MarTa-v2
# MarTa-v2
# marta-v2
