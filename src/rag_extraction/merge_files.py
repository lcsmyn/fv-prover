from string import punctuation
import json
import os
import re
import shutil
import gc
import sys

#put files partial names as args, for example: 
# python3 merge_files.py '0, 400' '400, 410' '410, 450'
### '0, 400' space at the middle
if __name__ == '__main__':

    train_total = 'train_rag_lemmas_total.json'

    with open('train_rag_lemmas_['+sys.argv[1:][0]+').json', 'r') as f1:
        merged_list = json.load(f1)
    for index in range(1, len(sys.argv[1:])):
        with open('train_rag_lemmas_['+sys.argv[1:][index]+').json', 'r') as f2:
            list_data2 = json.load(f2)
        # Concatenate the lists
        merged_list = merged_list + list_data2


    with open(train_total, 'w') as f:
        f.write(json.dumps(merged_list, indent=2))
    
    val_total = 'val_rag_lemmas_total.json'

    with open('val_rag_lemmas_['+sys.argv[1:][0]+').json', 'r') as f1:
        merged_list = json.load(f1)
    for index in range(1, len(sys.argv[1:])):
        with open('val_rag_lemmas_['+sys.argv[1:][index]+').json', 'r') as f2:
            list_data2 = json.load(f2)
        # Concatenate the lists
        merged_list = merged_list + list_data2


    with open(val_total, 'w') as f:
        f.write(json.dumps(merged_list, indent=2))
