# FV-Prover

This is the code for Lucas Yuan's 2025 Non-Trivial Fellowship Project.  Note: the following evaluation process will likely not work -- I finished this code a very short time before the deadline, so I had little time to clean up the repository and properly document.  Most of the files here are copied from other repositories (LeanDojo and FVEL) and are in various stages of completion.  The relevant files for the evaluation are src/retrieval_small/model.py, post.py, and isaCheck.py.

For (incomplete) setup details about this project, look at README_old.md.

## Project Overview

 Formal verification (FV) is a way to prove the correctness of code. FV treats a property of software as a mathematical theorem and applies logical steps to prove it. I applied LLMs to FV. I partially implemented a best-first proof search with retrieval-augmented generation on possible logical steps. Due to time constraints, I evaluated an LLM's ability to generate a formal proof of the termination of a program in one pass. The LLM had a 51% accuracy rate in proof generation. 

## Dataset Processing

I used the FVELER dataset in src/rag_extraction (this module of the project did not make it into the final evaluation).  If you want to recreate the datasets I made, simply unzip [FVELER](https://github.com/FVELER/FVELerExtraction/blob/main/FVELer.zip) to src/rag_extraction.  You can do this with the following commands, assuming you are in src/rag_extraction.
```
wget https://github.com/FVELER/FVELerExtraction/raw/refs/heads/main/FVELer.zip
unzip FVELer.zip
```
Then run
```
python3 -m src.rag_extraction.gen_dataset <begin> <end>
```
to choose how many theories you want to process.  begin = 0 and end = 100 usually works well.  Processing the entire FVELER dataset takes a very long time and is best done by running multiple python processes at once using tmux.

## Evaluation

Go to fv-prover root and run

```
python3 -m src.retrieval_small.model
python3 -m post
python3 -m isaCheck
```

to get the result of the evaluation in a large log file in fv-prover root.

Due to time constraints, I simply used grep to search for log messages that showed a pass or fail in Isabelle theory compilation.  I piped these messages to a new file and I counted how many of them passed to see the success rate of the LLM.