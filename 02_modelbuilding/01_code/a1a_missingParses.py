# -*- coding: utf-8 -*-
import a2_parsers, c2_utils, a1_dataloader
import dill, glob 
from nltk import sent_tokenize

### To Do:
# 1. finalise steps 2, 10, 11 below. (after c2_utils._make_wordsattr is finalised)

if __name__ == "__main__":
    
    LANG = 'en'
    DEP_FORMAT = 'SD'
    USAGE = 'experiments'
    
    for dataset in ['train', 'dev', 'test']:
        with open('./03_data/{}/pdtb_conll_data/ParsePDTB_dict_{}.dill'.format(LANG,dataset), 'rb') as f: 
            Parse_dict = dill.load(f)
        
        # get the list of wsj_nums that we don't have the parses for. 
        _available = set(Parse_dict.keys())
        raw = glob.glob('./03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/raw/*'.format(LANG, dataset))
        _raw = set([i.split('/')[-1] for i in raw])
        missing = _raw.difference(_available)

        # 
        for eachMissing in missing: 
            # 1. open the corresponding raw text file
            with open('./03_data/{}/pdtb_conll_data/conll16st-en-03-29-16-{}/raw/{}'.format(LANG, dataset, eachMissing)) as f:
                document = f.read()

            # 2. send to segmenter and tokeniser
            sentwords_list = c2_utils._make_wordsattr(document, usage=USAGE)
            
            _sent_list = sent_tokenize(document.lstrip('.START \n\n'))

            assert len(_sent_list) == len(sentwords_list)

            _sentences = list()
            _ptbparses = list()
            for sent_num in range(len(sentwords_list)):
                # 3. create a new Parse-class object 
                Parse = a1_dataloader.ParsePDTB(language = LANG, docid = eachMissing, pdtb_version=2)
                
                # 4. store sentword under the .Word attribute of the ParsePDTB object. 
                Parse.Words = sentwords_list[sent_num]

                # 5. get the PDTB parse for the sentence
                sentence =  sentwords_list[sent_num]
                Parse._ConstTree = a2_parsers._parse_rawtext(LANG, sentence, 
                                    berkparser_jarpath = '/04_utils/berkeleyparser-master/', tokenized=False) 
                _ptbparses.append(Parse._ConstTree)
 

                # 6. append the Parse object to the list of Parses for the sentence. 
                _sentences.append(Parse)
                

            # 7. add this list to the Parse_dict
            Parse_dict[eachMissing] = _sentences

            # 8. write the list of _ptbparses to file for SD/UD processing 
            with open('./03_data/{}/stree_parses/missing/{}_stree_PTB/'.format(LANG, dataset)+eachMissing, 'w+') as f: 
                [f.write('\n##&&##_'+str(i)+'\n'+_ptbparses[i]) for i in range(len(_ptbparses))]

            # 9. convert PTB parses for the sentence to UD parses 
            Parse._Dependencies = a2_parsers.convert_const2dep(lang=LANG, dataset=dataset, 
                                filename= eachMissing, readpath='./03_data/{}/stree_parses/missing/{}_stree_PTB/'.format(LANG, 
                                dataset)+eachMissing, writepath = './03_data/{}/stree_parses/missing/{}_stree_{}/'.format(LANG, 
                                dataset, DEP_FORMAT))

            # 10. Read the SD/UD CoNLL files and write to .DepTree_.. attribute for the relevant Parse-class object
            with open('./03_data/{}/stree_parses/missing/{}_stree_{}/'.format(LANG, dataset, DEP_FORMAT)) as f:
                depparses = f.read()

            _depparses_list
            while len(depparses) > 0: 
                _oneparse = list()
                _curr = depparses.pop(0)
                if _curr != '': 
                    deprel = _curr.split('\t')[7]
                    governor = _curr.split('\t')[6]
                    dependent = _curr.split('\t')[0]
                    _oneparse.append()
                else: 
                    pass

            Parse_dict[eachMissing][sent_num].DepTree_UD1_Auto = 

        # 11. save the new Parse_dict 
        with open('./03_data/{}/pdtb_conll_data/ParsePDTB_dict_with{}missing_{}.dill'.format(LANG, dataset, 
            DEP_FORMAT),'wb') as f:
            dill.dump(Parse_dict, f)
