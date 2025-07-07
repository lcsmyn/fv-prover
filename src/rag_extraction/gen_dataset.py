from string import punctuation
import json
import os
import re
import shutil
import gc
from tqdm import tqdm

def process_thy(thy):
    '''returns list of dictionaries containing processed
    proof steps and proof states in the theory'''
    # initialize theory-level variables
    lemmas = thy.get("lemmas")
    processed = []
    imported_terms = {}
    tokenizer = IsabelleTokenizer()

    # tokenize individual lemmas, replace local variables, replace
    # imported terms, concatenate tokenized lemmas to re-form original,
    # add processed state-step pairs to processed
    for lemma_num in range(lemmas.len()):

        # initialize lemma-level variables
        lemma = lemmas[lemma_num]
        proof = lemma.get("proof")
        proof_state = lemma.get("proof_state")

        # initialize proof state and step arrays for this lemma.
        # each state is an array.  string elems of the array are
        # proof-state formatting.  array elems are tokenized subgoals.
        tokenized_proof_states = []
        # each proof step is a tokenized array
        tokenized_proof_steps = []

        # tokenize all proof states and proof steps
        for index in range(1, proof.len()):
            tokenized_state = tokenizer.tokenize_state(proof_state[index-1])
            tokenized_step = tokenizer.tokenize(proof[index])
            tokenized_proof_states.append(tokenized_state)
            tokenized_proof_steps.append(tokenized_step)
            
        # replace local variables and extract imported terms from
        # all the tokenized proof states and steps of this lemma
        for i in range(tokenized_proof_states.len()):
            state = tokenized_proof_states[i]
            for j in range(state.len()):
                subgoal = state[j]
                if isinstance(subgoal, list):
                    subgoal = replace_local_vars(subgoal)
                    subgoal = replace_imported_terms(imported_terms, subgoal)
            step = tokenized_proof_steps[i]
            step = replace_local_vars(step)
            subgoal = replace_imported_terms(imported_terms, step)

        # concatenate states and steps and put them in processed
        for i in range(tokenized_proof_states.len()):
            state_string = ""
            state = tokenized_proof_states[i]
            for j in range(state.len()):
                part = state[j]
                if isinstance(part, str):
                    state_string += part
                else:
                    state_string += "".join(part)

            step_string = "".join(tokenized_proof_steps[i])

            processed.append({
                "state" : state_string,
                "step" : step_string
            })
    
    return processed

def process(text):
    