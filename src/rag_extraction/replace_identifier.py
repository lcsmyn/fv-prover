from string import punctuation
import json
import os
import re
import shutil
import gc
from tqdm import tqdm
from src.rag_extraction.isabelle_keywords import keyword_list
from src.rag_extraction.define import define

# handle fact info in the proof state later
class ThyIdReplacer():
    long_id_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9_\']{4,}')
    loc_var_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9_\']{0,3}')
    proof_state_pattern = re.compile(r'(?<=[0-9]+\.).*(?=[0-9]+\.)', re.DOTALL)
    isabelle_keywords = keyword_list
    global_dict = {}
    
    def __init__(self, import_dict):
        self.import_dict = import_dict

    def add_terms(self, string):
        terms = re.findall(ThyIdReplacer.long_id_pattern, string)
        
        for term in terms:
            if term not in ThyIdReplacer.isabelle_keywords:
                exact_term = r'(?<![a-zA-Z])' + term + r'(?![a-zA-Z0-9_\'])'
                compiled = re.compile(exact_term)

                if compiled not in ThyIdReplacer.global_dict:
                    term_def = define(compiled)
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
    
    def replace_loc_vars(self, string):
        return_string = str(string)
        return_string = re.sub(ThyIdReplacer.loc_var_pattern, '<|local|>', string)
        return return_string


    
                
