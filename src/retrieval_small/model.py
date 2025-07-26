import json
import faiss
import numpy as np

from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RetrievalAugmentedGenerator():
    def __init__(
            self,
            train_dataset_path,
            val_dataset_path,
            tokenizer_ckpt="sentence-transformers/all-MiniLM-L6-v2",
            embedding_model_ckpt="sentence-transformers/all-MiniLM-L6-v2",
            generator_ckpt="put in later",
            max_length=128,
            num_retrieved=10,
            warmup_steps=100,
            epochs=3
    ) -> None:
        self.train_dataset = json.load(open(train_dataset_path, "r"))
        self.val_dataset = json.load(open(val_dataset_path, "r"))
        self.indexed_steps = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        self.embedding_model = SentenceTransformer(embedding_model_ckpt)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_ckpt)
        self.max_length = max_length
        self.num_retrieved = num_retrieved
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
                max_length=128)
            helpful_step = self.tokenizer(
                step["step"],
                truncation=True,
                padding="max_length",
                max_length=128)
            
            # implement negative examples later.  First want 
            # a proof of concept

            # unhelpful_step = tokenizer(
            #     step["unrelated_state"],
            #     truncation=True,
            #     padding="max_length",
            #     max_length=128)
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

    def evaluate_embedding(self):
        states = {}
        steps = {}
        relevant_steps = {}

        n = 1
        for pair in self.val_dataset:
            states["state" + n] = pair.get("state")
            steps["step" + n] = pair.get("step")
            states["state" + n] = ["step" + n]

        # Initialize evaluator
        evaluator = InformationRetrievalEvaluator(states, steps, relevant_steps)

        # Evaluate the model
        self.embedding_model.evaluate(evaluator)

    def embed(self):
        embeddings = np.array([])
        self.indexed_steps = faiss.IndexFlatL2(384)

        for pair in self.train_dataset:
            tensor = self.embedding_model.encode_document(pair.get("step"))
            vec = tensor.numpy()
            np.concat((embeddings, [vec]), axis=0)

        self.indexed_steps.add(embeddings)

        print("embeddings finished")

    def retrieve(self, query):
        query_embedding = self.embedding_model.encode_query(query)
        _, indices = self.indexed_steps.search(query_embedding, k=self.num_retrieved)
        steps = []
        for index in indices:
            steps.append(self.train_dataset[index].get("step"))
        return steps
    
    def generate_tacs(self, proof_state):
        retrieved_tacs = self.retrieve(proof_state)
        

        tokenized_state = self.tokenizer(
            proof_state, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        state_ids = tokenized_state.input_ids
        state_mask = tokenized_state.attention_mask
        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_oup_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
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

        for j in range(num_samples):
            t = remove_marks(raw_output_text[j])
            if self.decoder_only and t.startswith(state):
                t = t[len(state) :]
            if t not in output_text:
                output_text.append(t)
                output_score.append(raw_scores[j])

        return list(zip_strict(output_text, output_score))