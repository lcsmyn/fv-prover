from string import punctuation
import json
import os
import re
import shutil
import gc
from tqdm import tqdm
from isabelle_keywords import keywords

# handle fact info in the proof state later
# also consider the (prove), (state), and (chain) things.  I don't know what they are
# actually, this seems to be an easy fix just by matching "this:" in the proof state
class ThyIdReplacer():
    long_id_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9_\']{3,}')
    loc_var_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9_\']{0,2}')
    proof_state_pattern = re.compile(r'(?:[0-9]\.|this:).*(?=[0-9]+\.|$)', re.DOTALL)
    isabelle_keywords = keywords
    global_dict = {}
    
    def __init__(self, import_dict):
        self.import_dict = import_dict

    def add_terms(self, string):
        term_matches = re.finditer(ThyIdReplacer.long_id_pattern, string)
        
        for match in term_matches:
            start_index = match.start()
            term = match.group()
            if (start_index < 2) or (string[start_index-2:start_index] == r'\<'): # handle escape sequences
                continue
            elif term not in ThyIdReplacer.isabelle_keywords:
                exact_term = r'(?<![a-zA-Z])' + term + r'(?![a-zA-Z0-9_\'])'
                compiled = re.compile(exact_term)

                if compiled not in ThyIdReplacer.global_dict:
                    term_def = "<|long_id|>"
                    ThyIdReplacer.global_dict[compiled] = term_def
                    self.import_dict[compiled] = term_def
                elif compiled not in self.import_dict:
                    self.import_dict[compiled] = ThyIdReplacer.global_dict[compiled]
    
    def add_state_terms(self, state_string):
        goals = re.findall(ThyIdReplacer.proof_state_pattern, state_string)

        for subgoal in goals:
            self.add_terms(subgoal)

    # Check if this works for proof state as well
    def replace(self, string):
        return_string = str(string)

        for pattern, repl in self.import_dict.items():
            return_string = re.sub(pattern, repl, return_string)

        return return_string
    
    # def replace_loc_vars(self, string):
    #     return_string = str(string)
    #     return_string = re.sub(ThyIdReplacer.loc_var_pattern, '<|local|>', string)
    #     return return_string


    
                
