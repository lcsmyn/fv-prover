from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import json
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small") # check if this is the right one
model = SentenceTransformer("kaiyuy/leandojo-lean4-retriever-byt5-small") # check if right

print("tokenizer and model created")

with open("put json file here", "r") as f:
    dataset = json.dump(f)

def preprocess(step):
    query_state = tokenizer(
        step["state"],
        truncation=True,
        padding="max_length",
        max_length=128)
    helpful_step = tokenizer(
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
    return [
        InputExample(texts=[query_state, helpful_step], label=0.9)
        ]

train_examples = []

for step in dataset:
    train_examples.extend(preprocess(step))

print("data processed successfully")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

print("fine-tuned succesfully")

# Define queries, documents, and relevant pairs
states = {}
steps = {}
relevant_steps = {}

n = 1
for pair in f:
    states["state" + n] = pair.get("state")
    steps["step" + n] = pair.get("step")
    states["state" + n] = ["step" + n]

# Initialize evaluator
evaluator = InformationRetrievalEvaluator(states, steps, relevant_steps)

# Evaluate the model
model.evaluate(evaluator)