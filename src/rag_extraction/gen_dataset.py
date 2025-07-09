from string import punctuation
import json
import os
import re
import shutil
import gc
from tqdm import tqdm

from replace_identifier import *

def process_thy(thy):
    '''returns list of dictionaries containing processed
    proof steps and proof states in the theory'''
    # initialize theory-level variables
    lemmas = thy.get("lemmas")
    processed = []
    # initialize the replacer here for best efficiency?
    replacer = ThyIdReplacer({})

    # get identifiers of individual lemmas, replace local variables, replace
    # imported terms, concatenate processed lemmas to re-form original,
    # add processed state-step pairs to processed
    if lemmas is None:
        return []

    for lemma in lemmas:
        # initialize lemma-level variables
        proof_state = lemma.get("proof_state")
        proof = lemma.get("proof")

        for index in range(1, len(proof)):
            state = proof_state[index-1]
            step = proof[index]

            replacer.add_state_terms(state)
            replacer.add_terms(step)

            new_state = replacer.replace(state)
            new_state = replacer.replace(step)

            new_state = replacer.replace_loc_vars(state)
            new_step = replacer.replace_loc_vars(step)

            processed.append({
                "state" : new_state,
                "step" : new_step
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
    gc.collect()

    return processed
    
def process_dataset():
    with open('sel4_thy_info.json', 'r') as f:
        original_dataset = json.load(f)

    processed_dataset = []

    for name, theory in tqdm(original_dataset.items()):
        theory_states_steps = process_thy(theory)
        for pair in theory_states_steps:
            processed_dataset.append(pair)
        # print("Processed " + name)
    
    return processed_dataset

if __name__ == '__main__':
    data = process_dataset()
    
    with open('dataset_rag_lemmas.json', 'w') as f:
        f = json.dump(data)