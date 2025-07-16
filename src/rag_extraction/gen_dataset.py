from string import punctuation
import json
import os
import re
import shutil
import gc
from tqdm import tqdm
from copy import deepcopy
import sys

from replace_identifier import *

# TODO: add negative examples (based on dependency graph distance) and
# create train/val splits.

def process_thy(thy):
    '''returns list of dictionaries containing processed
    proof steps and proof states in the theory'''
    # initialize theory-level variables
    thy_name = thy[0]
    thy_info = thy[1]
    lemmas = thy_info.get("lemmas")
    processed_train = []
    processed_val = []
    # initialize the replacer here for best efficiency?
    replacer = ThyIdReplacer({}, thy_name)

    # get identifiers of individual lemmas, replace local variables, replace
    # imported terms, concatenate processed lemmas to re-form original,
    # add processed state-step pairs to processed
    if lemmas is None:
        pass

    #print("lemmas: ", len(lemmas))
    #total = 0
    for index in len(lemmas):
        lemma = lemmas[index]
        # initialize lemma-level variables
        proof_state = lemma.get("proof_state")
        proof = lemma.get("proof")

        #total = total + len(proof)
        for index in range(1, len(proof)):
            print(".", end="", flush=True)
            state = proof_state[index-1]
            step = proof[index]

            replacer.add_state_terms(state)
            replacer.add_terms(step)

            new_state = replacer.replace(state)
            new_step = replacer.replace(step)

            # new_state = replacer.replace_loc_vars(state)
            # new_step = replacer.replace_loc_vars(step)
            if (index % 3 == 1):       
                processed_val.append({
                    "state" : new_state,
                    "step" : new_step,
                    "original_state" : state,
                    "original_step" : step
                })
            else:
                processed_train.append({
                    "state" : new_state,
                    "step" : new_step,
                    "original_state" : state,
                    "original_step" : step
                })

        # # initialize proof state and step arrays for this lemma.
        # # each state is an array.  string elems of the array are
        # # proof-state formatting.  array elems are subgoal substrings
        # # with identifiers being single elements
        # split_proof_states = []
        # # each proof step is an array of identifiers or surrounding code
        # split_proof_steps = []

        # # split all proof states and proof steps
        # for index in range(1, proof.len()):
        #     split_state = id_finder.find_id_state(proof_state[index-1])
        #     split_step = id_finder.find_id(proof[index])
        #     split_proof_states.append(split_state)
        #     split_proof_steps.append(split_step)
            
        # # replace local variables and extract imported terms from
        # # all the split proof states and steps of this lemma
        # for i in range(split_proof_states.len()):
        #     state = split_proof_states[i]
        #     for j in range(state.len()):
        #         subgoal = state[j]
        #         if isinstance(subgoal, list):
        #             subgoal = replace_local_vars(subgoal)
        #             imported_terms = add_imported_terms(imported_terms, subgoal, thy)
        #             subgoal = replace_imported_terms(imported_terms, subgoal)
        #     step = split_proof_steps[i]
        #     step = replace_local_vars(step)
        #     imported_terms = add_imported_terms(imported_terms, subgoal, thy)
        #     step = replace_local_vars(step)

        # # concatenate states and steps and put them in processed
        # for i in range(split_proof_states.len()):
        #     state_string = ""
        #     state = split_proof_states[i]
        #     for j in range(state.len()):
        #         part = state[j]
        #         if isinstance(part, str):
        #             state_string += part
        #         else:
        #             state_string += "".join(part)

        #     step_string = "".join(split_proof_steps[i])

        #     processed.append({
        #         "state" : state_string,
        #         "step" : step_string
        #     })
    #print("This total: ", total)
    gc.collect()

    return processed
    
def process_dataset(begin, end):
    with open('sel4_thy_info.json', 'r') as f:
        original_dataset = json.load(f)



    processed_dataset = []

    #for name, theory in original_dataset.items():
    index = 0
    for theory in tqdm(original_dataset.items()):    
        if (index >= begin and index < end):
            theory_states_steps = process_thy(theory)
            for pair in theory_states_steps:
                processed_dataset.append(pair)

            #print("Processed " + name)
        index += 1
    
    return processed_dataset

if __name__ == '__main__':
    begin = int(sys.argv[1:][0])
    end = int(sys.argv[1:][1])

    data = process_dataset(begin, end)

    fname = 'dataset_rag_lemmas_['+str(begin)+', '+str(end)+').json'
    with open(fname, 'w') as f:
        f.write(json.dumps(data, indent=2))
    # if not os.path.isfile(fname):
    #     with open(fname, 'w') as f:
    #         f.write(json.dumps(data, indent=2))
    # else:
    #     with open(fname) as feedsjson:
    #         feeds = json.load(feedsjson)

    #     feeds.append(data)
    #     with open(fname, mode='w') as f:
    #         f.write(json.dumps(feeds, indent=2))


    #with open('dataset_rag_lemmas.json', 'w') as f:
    #    f = json.dump(data, f)