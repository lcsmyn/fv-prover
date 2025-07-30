# FV-Prover

This is the code for Lucas Yuan's 2025 Non-Trivial Fellowship Project.  Note: the following setup and evaluation process will likely not work -- I finished this code a very short time before the deadline, so I had little time to clean up the repository and properly document.  Most of the files here are copied from other repositories (LeanDojo and FVEL) and are in various stages of completion.  The relevant files for the evaluation are src/retrieval_small/model.py, post.py, and isaCheck.py

## setup

Here is the setup process, tested for Ubuntu 22.04.

### 1. Install Isabelle2022.

```bash=
# for ARM machines
wget https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux_arm.tar.gz 
tar -xzvf Isabelle2022_linux_arm.tar.gz
# for x86_64 machines
wget https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux.tar.gz 
tar -xzvf Isabelle2022_linux.tar.gz
# It is recommended to write the following two lines to ~/.bashrc (or to ~./zshrc)
export PATH="$PATH:$HOME/Isabelle2022/bin"
export ISABELLE_HOME="$HOME/Isabelle2022"
```


You can test the installation by typing:

```bash=
which isabelle
```

and it's expected to output:

```bash
$PATH_TO_YOUR_HOME/Isabelle2022/bin/isabelle
```

Tips: If your system supports graphics, you can type the following command to open the Isabelle JEditor:

```bash
$PATH_TO_YOUR_HOME/Isabelle2022/Isabelle2022
```

### 2. Clone the l4v repo.

NOTE: We do not recommend tracking the latest l4v repo due to some compatibility issues.

```bash=
git clone https://github.com/FVELER/l4v-FVEL
cd l4v-FVEL
ln -s $ISABELLE_HOME ./isabelle
```

### 3. Build dependency for l4v proof.
Follow the guidance in [setup.md](https://github.com/FVELER/l4v-FVEL/blob/main/docs/setup.md) to build dependency for l4v proof. The minimal dependency includes ``texlive`` (to build document) and [mlton-compiler](http://www.mlton.org/Home).

In addition to the dependencies in setup.md, install gmp, mlton, and polyml.

Install the Python dependency:
```bash=
pip install --user sel4-deps
```

### 4. Install C-parser and Auto-corres.

Run the following commands:
```bash=
# It is recommended to write the following line to ~/.bashrc
export L4V_ARCH=ARM # Choices: [ARM, X86, X64]
cd $PATH_TO_L4V/tools/c-parser
isabelle env make
cd $PATH_TO_L4V
isabelle build -d . CParser
cd $PATH_TO_L4V/
./run_tests AutoCorres
```

It's expected to output:
```
3/3 tests succeeded.

All tests passed.
```

### 5. Unzip the FVELER dataset

I used the FVELER dataset in src/rag_extraction (this module of the project did not make it into the final evaluation).  If you want to recreate the datasets I made, simply unzip [FVELER](https://github.com/FVELER/FVELerExtraction/blob/main/FVELer.zip) to src/rag_extraction.  Then run
```
python3 -m src.rag_extraction.gen_dataset <begin> <end>
```
to choose how many theories you want to process.  begin = 0 and end = 100 usually works well.  Processing the entire FVELER dataset takes a very long time, though. 

## Evaluation

Go to fv-prover root and run

```
python3 -m src.retrieval_small.model
python3 -m post
python3 -m isaCheck
```

to get the result of the evaluation in a large log file in fv-prover root.

Due to time constraints, I simply used grep to search for log messages that showed a pass or fail in Isabelle theory compilation.  I piped these messages to a new file and I counted how many of them passed to see the success rate of the LLM.