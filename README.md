# FVEL
Code for implement of paper: [FVEL: Interactive Formal Verification Environment with Large Language Models via Theorem Proving](https://arxiv.org/abs/2406.14408).

## setup

Here is the setup process for FVEL framework. (Tested on Ubuntu-20.04 and Debian.)

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

### 5. Install Portal-to-ISAbelle.
We maintain a l4v-compatible [Portal-to-ISAbelle](https://github.com/albertqjiang/Portal-to-ISAbelle) repo in [PISA_FVEL](https://github.com/FVELER/PISA_FVEL). Run the following commands to install PISA_FVEL:

```bash=
git clone https://github.com/FVELER/PISA_FVEL.git

# install sdkman
curl -s "https://get.sdkman.io" | bash
source ~/.bashrc
# to make sure sdk is properly installed:
sdk help

# install java and sbt
sdk install java 11.0.11-open
sdk install sbt

cd PISA_FVEL
sbt compile
```

When the installation is complete, run the server at PISA root path to prove in one pass:

```bash=
sbt "runMain pisa.server.PisaOneStageServer8000"    # 8000 is the port exposed to the client
```
