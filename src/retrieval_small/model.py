import json
import faiss
import pickle
import numpy as np
import torch
import tqdm
import gc
import os

from typing import Optional
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class RetrievalAugmentedGenerator():
    def __init__(
            self,
            device,
            train_dataset_path=None,
            val_dataset_path=None,
            indexed_steps_path=None,
            tokenizer_ckpt="DeepSeek-Prover-V2-7B",
            embedding_model_ckpt="Qwen3-Embedding-0.6B",
            generator_ckpt="DeepSeek-Prover-V2-7B",
            max_state_inp_length=1024,
            max_tacgen_inp_length=1024,
            max_new_tokens=32000,
            num_retrieved=10,
            num_samples=1,
            warmup_steps=100,
            epochs=3
    ) -> None:
        self.device = device

        if train_dataset_path:
            self.train_dataset = json.load(open(train_dataset_path, "r"))
        else:
            self.train_dataset = None

        if val_dataset_path:
            self.val_dataset = json.load(open(val_dataset_path, "r"))
        else:
            self.val_dataset = None

        if indexed_steps_path:
            self.indexed_steps = pickle.load(open(indexed_steps_path, "rb"))
        else:
            self.indexed_steps = None
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
     
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)

        if embedding_model_ckpt:
            self.embedding_model = SentenceTransformer(embedding_model_ckpt).to(self.device)
            self.embedding_model_name = embedding_model_ckpt
        else:
            self.embedding_model = None
            self.embedding_model_name = None

        if generator_ckpt:
            self.generator = AutoModelForCausalLM.from_pretrained(generator_ckpt, quantization_config=quantization_config) 
        else: # i.e. training embedding model, don't need generator
            self.generator = None

        self.max_state_inp_length = max_state_inp_length
        self.max_tacgen_inp_length = max_tacgen_inp_length
        self.max_new_tokens = max_new_tokens
        self.num_retrieved = num_retrieved
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        print("models loaded successfully")

    def preprocess_data(self, data):
        train_examples = []
        for step in data:
            query_state = self.tokenizer(
                step["state"],
                truncation=True,
                padding="max_length",
                max_length=self.max_state_inp_length)
            helpful_step = self.tokenizer(
                step["step"],
                truncation=True,
                padding="max_length",
                max_length=self.max_state_inp_length) # yes, the variable is confusingly named
            
            # implement negative examples later.  First want 
            # a proof of concept

            # unhelpful_step = tokenizer(
            #     step["unrelated_state"],
            #     truncation=True,
            #     padding="max_length",
            #     max_length=self.max_state_inp_length)
            ex = [InputExample(texts=[query_state, helpful_step], label=0.9)]
            train_examples.extend(ex)

        print("preprocessed successfully")

        return train_examples

    # Note: model.fit is deprecated
    def train_embedding(self):
        train_dataloader = DataLoader(self.preprocess_data(self.train_dataset), shuffle=True, batch_size=2)
        train_loss = losses.CosineSimilarityLoss(self.embedding_model)
        torch.cuda.empty_cache()
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluator=None,
            use_amp=True
        )
        print("fine-tuning finished")
        self.embedding_model.save_pretrained("./checkpoint_embedding_" + self.embedding_model_name)

    def evaluate_embedding(self):
        states = {}
        steps = {}
        relevant_steps = {}

        n = 1
        for pair in self.val_dataset:
            state = pair.get("state")
            step = pair.get("step")
            if state.strip() and step.strip():
                states["state" + str(n)] = state
                steps["step" + str(n)] = step
                relevant_steps["state" + str(n)] = ["step" + str(n)]

        # Initialize evaluator
        evaluator = InformationRetrievalEvaluator(states, steps, relevant_steps)

        # Evaluate the model
        print(self.embedding_model.evaluate(evaluator))

    def embed(self):
        self.indexed_steps = faiss.IndexFlatL2(384)

        for i in tqdm(range(len(self.train_dataset))):
            tensor = self.embedding_model.encode_document(self.train_dataset[i].get("step"))
            if i == 0:
                embeddings = np.array([tensor])
            else:
                embeddings = np.concatenate((embeddings, [tensor]), axis=0)

        self.indexed_steps.add(embeddings)
        with open('indexed_steps_'+self.embedding_model_name+'.pickle', 'wb') as f:
            pickle.dump(self.indexed_steps, f)

        print(f"Vectors in index: {self.indexed_steps.ntotal}")
        print("embeddings finished")

    def retrieve(self, query):
        query_embedding = self.embedding_model.encode_query(query)
        _, indices = self.indexed_steps.search(query_embedding.reshape(1, -1), k=self.num_retrieved)
        steps = []
        for index in indices.tolist()[0]:
            steps.append(self.train_dataset[index].get("step"))
        print(steps)
        return steps
    
    def format_state(self, proof_state, tacs):
        instr1 = "Isabelle proof state:\n"
        # idea: maybe add all previous proof states as well.
        # instr2 = "\ngenerate a single proof step that makes progress in the proof.  Ensure that the step is logically sound \
        #           and free of redundant content.  Use appropriate tactics and lemmas as necessary.  Don't explain.  Here are some \
        #           useful proof step structures you can use (<|long_id|> represents some identifier, like a variable or function):\n"
        instr2 = "Possible tactic structures (<|long_id|> is any identifier):\n"
        instr3 = "Next tactic:"
        new_state = instr1
        new_state += (proof_state + "\n")
        new_state += instr2
        for tac in tacs:
            new_state += (tac + "\n")
        new_state += instr3

        return new_state
    
    def format_1pass(self, isa_file_path, c_file_path):
        with open(isa_file_path, "r") as f:
            isa_file = f.read()
        with open(c_file_path, "r") as f:
            c_file = f.read()

        instr1 = f"Here is a program in C called {c_file_path}:\n"
        instr2 = "Here is a theory file in Isabelle about this program:\n"
        instr3 = "The theorem in the Isabelle file aims to prove that the C program terminates.  If the program does not terminate, output the single word 'False'.\
                  If it does terminate, write the proof for the theorem.  Ensure that the proof is complete, logically sound and free of redundant content. \
                  Use appropriate tactics and lemmas as necessary. Don’t explain."
        return (instr1 + c_file + instr2 + isa_file + instr3)

    def generate_1pass(self, isa_file_path, c_file_path):
        prompt = self.format_1pass(isa_file_path, c_file_path)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tacgen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(formatted_prompt)

        tokenized_state = tacgen.tokenizer(
            formatted_prompt,
            max_length=tacgen.max_tacgen_inp_length, 
            truncation=True, 
            return_tensors="pt"
        )
        prompt_ids = tokenized_state.input_ids.to(self.device)
        input_length = prompt_ids.shape[1]

        prompt_mask = tokenized_state.attention_mask.to(self.device)
        # Generate tactic candidates using beam search.
        output = tacgen.generator.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.2,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_sequences = output.sequences[0][input_length:]

        # Return the output.
        raw_output_text = tacgen.tokenizer.decode(
            generated_sequences, skip_special_tokens=True
        )
        return raw_output_text
    
    def generate_tacs(self, proof_state):
        # TODO: make sure to process the proof_state here
        retrieved_tacs = self.retrieve(proof_state)

        # tokenized_state = self.tokenizer(
        #     self.format_state(proof_state, retrieved_tacs), max_length=self.max_tacgen_inp_length, truncation=True, return_tensors="pt"
        # )
        # state_ids = tokenized_state.input_ids.to('cuda')
        # state_mask = tokenized_state.attention_mask.to('cuda')
        # # Generate tactic candidates using beam search.
        # output = self.generator.generate(
        #     input_ids=state_ids,
        #     attention_mask=state_mask,
        #     max_length=self.max_new_tokens,
        #     num_beams=self.num_samples,
        #     # length_penalty=self.length_penalty,
        #     do_sample=False,
        #     num_return_sequences=self.num_samples,
        #     early_stopping=False,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        # )

        # # Return the output.
        # raw_output_text = self.tokenizer.batch_decode(
        #     output.sequences, skip_special_tokens=True
        # )
        # raw_scores = output.sequences_scores.tolist()

        # output_text = []
        # output_score = []

        # for j in range(self.num_samples):
        #     t = raw_output_text[j]
        #     if t not in output_text:
        #         output_text.append(t)
        #         output_score.append(raw_scores[j])

        # return list(zip(output_text, output_score))
    
        messages = [{"role": "user", "content": self.format_state(proof_state, retrieved_tacs)}]

        formatted_prompt = tacgen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(formatted_prompt)

        tokenized_state = tacgen.tokenizer(
            formatted_prompt,
            max_length=tacgen.max_tacgen_inp_length, 
            truncation=True, 
            return_tensors="pt"
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        input_length = state_ids.shape[1]

        state_mask = tokenized_state.attention_mask.to(self.device)
        # Generate tactic candidates using beam search.
        output = tacgen.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_new_tokens=self.max_new_tokens,
            num_beams=5, #tacgen.num_samples,
            # length_penalty=self.length_penalty,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.2,
            num_return_sequences=tacgen.num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_sequences = output.sequences[:, input_length:]

        # Return the output.
        raw_output_text = tacgen.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()

        output_text = []
        output_score = []

        for j in range(tacgen.num_samples):
            t = raw_output_text[j]
            if t not in output_text:
                output_text.append(t)
                output_score.append(raw_scores[j])

        print(list(zip(output_text, output_score)))
    
if __name__ == "__main__":
    print(f"PyTorch sees: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    print(f"PyTorch cache: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])
    # Training script for the embedding model.

    tacgen = RetrievalAugmentedGenerator(
        device="cuda",
        tokenizer_ckpt="Qwen/Qwen3-8B",
        generator_ckpt="Qwen/Qwen3-8B"
    )
    print(tacgen.generate_1pass("jain_2-1.thy", "jain_2-1.c"))