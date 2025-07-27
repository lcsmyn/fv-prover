import json
import faiss
import pickle
import numpy as np
import torch
import gc

from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM

class RetrievalAugmentedGenerator():
    def __init__(
            self,
            train_dataset_path,
            val_dataset_path,
            indexed_steps_path=None,
            tokenizer_ckpt="Qwen/Qwen3-8B",
            embedding_model_ckpt="sentence-transformers/all-MiniLM-L6-v2",
            generator_ckpt="Qwen/Qwen3-8B", # Let's see what happens
            max_state_inp_length=128,
            max_tacgen_inp_length = 256,
            max_tacgen_oup_length=512,
            num_retrieved=10,
            num_samples=5,
            warmup_steps=100,
            epochs=3
    ) -> None:
        self.train_dataset = json.load(open(train_dataset_path, "r"))
        self.val_dataset = json.load(open(val_dataset_path, "r"))

        if indexed_steps_path:
            self.indexed_steps = pickle.load(open(indexed_steps_path, "rb"))
        else:
            self.indexed_steps = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        self.embedding_model = SentenceTransformer(embedding_model_ckpt).to('cuda')
        self.generator = AutoModelForCausalLM.from_pretrained(generator_ckpt, torch_dtype=torch.float16).cuda()
        self.max_state_inp_length = max_state_inp_length
        self.max_tacgen_inp_length = max_tacgen_inp_length
        self.max_tacgen_oup_length = max_tacgen_oup_length
        self.num_retrieved = num_retrieved
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.train_examples = []
        print("models loaded successfully")

    def preprocess_data(self):
        for step in self.train_dataset:
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
            self.train_examples.extend(ex)

        print("preprocessed successfully")
    
    # Note: model.fit is deprecated
    def train_embedding(self):
        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=32)
        train_loss = losses.CosineSimilarityLoss(self.embedding_model)
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps
        )
        print("fine-tuning finished")
        self.embedding_model.save_pretrained("./checkpoint_embedding")

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

        for i in range(len(self.train_dataset)):
            tensor = self.embedding_model.encode_document(self.train_dataset[i].get("step"))
            if i == 0:
                embeddings = np.array([tensor])
            else:
                embeddings = np.concatenate((embeddings, [tensor]), axis=0)

        self.indexed_steps.add(embeddings)
        with open('indexed_steps.pickle', 'wb') as f:
            pickle.dump(self.indexed_steps, f)

        print(f"Vectors in index: {self.indexed_steps.ntotal}")
        print("embeddings finished")

    def retrieve(self, query):
        query_embedding = self.embedding_model.encode_query(query)
        _, indices = self.indexed_steps.search(query_embedding.reshape(1, -1), k=self.num_retrieved)
        steps = []
        for index in indices.tolist()[0]:
            print(index)
            steps.append(self.train_dataset[index].get("step"))
        return steps
    
    def format_state(self, proof_state, tacs):
        instr1 = "You are an expert in Isabelle2022 formal software proofs.  Given the following proof state:\n"
        # idea: maybe add all previous proof states as well.
        instr2 = "\ngenerate a single proof step that makes progress in the proof.  Ensure that the step is logically sound \
                  and free of redundant content.  Use appropriate tactics and lemmas as necessary.  Don't explain.  Here are some \
                  useful proof step structures you can use (<|long_id|> represents some identifier, like a variable or function):\n"
        new_state = ""
        new_state += instr1
        new_state += (proof_state + "\n")
        new_state += instr2
        for tac in tacs:
            new_state += (tac + "\n")

        return new_state
    
    def generate_tacs(self, proof_state):
        retrieved_tacs = self.retrieve(proof_state)

        tokenized_state = self.tokenizer(
            self.format_state(proof_state, retrieved_tacs), max_length=self.max_tacgen_inp_length, truncation=True, return_tensors="pt"
        )
        state_ids = tokenized_state.input_ids.to('cuda')
        state_mask = tokenized_state.attention_mask.to('cuda')
        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_tacgen_oup_length,
            num_beams=self.num_samples,
            # length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=self.num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()

        output_text = []
        output_score = []

        for j in range(self.num_samples):
            t = raw_output_text[j]
            if t not in output_text:
                output_text.append(t)
                output_score.append(raw_scores[j])

        return list(zip(output_text, output_score))
    
if __name__ == "__main__":
    print(f"PyTorch sees: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
    print(f"PyTorch cache: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

    tacgen = RetrievalAugmentedGenerator(
        train_dataset_path="src/retrieval_small/train_rag_lemmas_[0, 100).json",
        val_dataset_path="src/retrieval_small/val_rag_lemmas_[0, 100).json",
        embedding_model_ckpt="checkpoint_embedding",
        indexed_steps_path="indexed_steps.pickle"
    )