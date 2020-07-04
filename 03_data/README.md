
### Converting PTB gold parses into Universal Dependencies v1.0 and Stanford Dependencies

The CoNLL 2015 and 2016 Shared Task datasets do not include Universal Dependencies part-of-speech tags, which are coarser-grained compared to the Penn Tree Bank part-of-speech tags. In order to obtain the UD-equivalent POS tags from the PTB parses, we utilise the Stanford CoreNLP package’s edu.stanford.nlp.trees.ud.UniversalDependenciesConverter class. To replicate our process, you will need the following:

1. Download CoreNLP https://stanfordnlp.github.io/CoreNLP/download.html (the stanford-corenlp-full-2018-10-05 version) and build with Maven or Ant. 
2. Extract the constituency parse (in S-tree form) from the CoNLL 2015 and 2016 Shared Task parse.json files.  A set of helper functions, including extract_PTBparses, can be found in our c2_utils module. The extract_PTBparses helps extracts PTB parses in S-tree format that have already been saved in a series of Parse-class objects. 
3. In command line, use the following command to:

a. convert the PTB parses to UD1.0 for the dev set 
</u>for file in _<relative path>_/marta-v2/03_data/en/stree_parses/dev_stree_PTB/*; do java -mx1g -cp "*" edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile $file > _<relative path>_/marta-v2/03_data/en/stree_parses/dev_conllu_UD1/${file##*/}; done    </u> 

b. convert the PTB parses to basic SD for the dev set 
</u>for file in _<relative path>_/marta-v2/03_data/en/stree_parses/train_stree_PTB/*; do java -mx1g -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure  -basic -conllx -treeFile $file > _<relative path>_/marta-v2/03_data/en/stree_parses/train_conllu_SD/${file##*/}; done     </u>

Replace “dev” with “train” or “test” throughout, in order to handle the other datasets. include the ‘-parse.originalDependencies’ argument in the java command in order to convert the POS tags to Stanford Dependencies format, if you require these. A helper function convert_ptb2ud in the c2_utils module does this programmatically.

Relevant instructions on the use of CoreNLP’s:

a. edu.stanford.nlp.trees.ud.UniversalDependenciesConverter Java class can be found [here](https://nlp.stanford.edu/software/stanford-dependencies.shtml) and [here](https://stanfordnlp.github.io/CoreNLP/cmdline.html).
 
b. edu.stanford.nlp.trees.EnglishGrammaticalStructure Java class can be found [here] (https://github.com/clulab/processors/wiki/Converting-from-Penn-Treebank-to-Basic-Stanford-Dependencies). Note that it is not necessarily to use CoreNLP 3.5.1 as stated on the page. Our experiments utilised CoreNLP 3.9.2 and minor randomised checks indicate coherence with the CoNLL 2016 Shared Task dependencies information. 