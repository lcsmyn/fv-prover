import os
import re

from .dsp_utils import Checker

def extract_fact(c_program_name: str, checker: Checker):
    func2fact =  checker.get_c_func_fact(c_program_name)
    return func2fact

def main():
    # quick workaround
    os.environ['PISA_PATH'] = "/var/home/richard/SourceCode/fv-prover/PISA_FVEL"

    l4v_dir = os.environ.get("L4V_DIR", "")
    isa_home = os.environ.get("ISABELLE_HOME", "")
    theory_file = "../../data/Interactive.thy"
    port = 8000
    checker = Checker(
        working_dir=f"AutoCorres;{l4v_dir}",
        isa_path=isa_home,
        theory_file=theory_file,
        port=port,
    )

    top = "../../data/normalized/code2inv"
    for root, _, files in os.walk(top):
        for file in files:
            if file.endswith(".c"):
                full_path = os.path.join(root, file)
                abs_path = os.path.abspath(full_path)
                func2fact = extract_fact(abs_path, checker)
                print(func2fact)
                input()

def normalize(c_file, output_dir="../../data/normalized/"):

    # replace assertions and assumptions
    assertion = "void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }\n"
    assumption = "void assume(int cond) { if (!cond) { abort(); } }\n"

    with open(c_file, "r", encoding="utf8") as f:
        code = f.read()
    code = code.replace(assertion, "").replace(assumption, "")
    code = re.sub(r'assert\((.*?)\);', r'if (not (\1)) {return -1;}', code)
    code = re.sub(r'assume\((.*?)\);', r'if (not (\1)) {return -1;}', code)

    # add external definitions
    if "unknown()" in code:
        code = "extern int unknown(void);\n" + code

    # remove redundant

    file_name = os.path.basename(c_file)
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, "w", encoding="utf8") as f:
        f.write(code)

    return code

if __name__ == "__main__":
    main()
    # output_dir = "./data/normalized/code2inv/"
    # for root, _, files in os.walk("./data/code2inv/c"):
    #     for file in files:
    #         if file.endswith(".c"):
    #             full_name = os.path.join(root, file)
    #             normalize(full_name, output_dir)