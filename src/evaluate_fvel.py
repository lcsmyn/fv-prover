import argparse
import hashlib
import json
import os
import re
import sys
import pickle
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from PISA_FVEL.src.main.python.pisa_client import Theorem
from prover.proof_search_fvel import CheatingProver, DistributedProver
# from prover.proof_search import DistributedProver, GPT4TacticGenerator
from prover.search_tree import Status


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
    begin_num: int,
    runed_logs: list[str],
) -> list[Theorem]:
    theorems = _isa_get_theorems_from_files(
            data_path,
            split,
            file_path,
            full_name,
            name_filter,
            num_theorems,
            begin_num,
            runed_logs,
    )
    return theorems

def _isa_get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
    begin_num: Optional[int],
    runed_logs: List[str],
) -> List[Theorem]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []

    finished_theorem = []
    if runed_logs is not None:
        assert begin_num is None, "runed_logs and begin_num cannot be used together."
        for log in runed_logs:
            logger.info(f"Processing the runed log: {log}")
            f = open(log, "r")
            for line in f:
                if "SearchResult" not in line:
                    continue
                pattern = r"Theorem\(file_path=PosixPath\('(.*?)'\), full_name=[\'\"](.*?)[\'|\"], count=(\d+)"
                match = re.search(pattern, line)
                assert match
                finished_theorem.append(Theorem(
                    file_path=Path(match.group(1)), 
                    full_name=match.group(2).encode().decode('unicode_escape'), 
                    count=int(match.group(3)))
                )
            f.close()
    logger.info(f"There are {len(finished_theorem)} theorem finished proving")

    for elem in data:
        if file_path is not None and elem["file_path"] != file_path:
            continue
        if full_name is not None and not elem["full_name"].startswith(full_name):
            continue
        if name_filter is not None and not hashlib.md5(
            elem["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        file_path = Path('/root/FVEL/env/Interactive.thy')
        work_dir = 'AutoCorres;/root/l4v_FVEL/'
        new_theorem = Theorem(file_path=file_path, full_name=elem["full_name"], count=elem["count"], working_directory=work_dir)
        if new_theorem not in finished_theorem:
            theorems.append(new_theorem)

    theorems = sorted(
        theorems,
        key=lambda t: hashlib.md5(
            (str(t.file_path) + ":" + t.full_name).encode()
        ).hexdigest(),
    )
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
    
    if begin_num is not None:
        theorems = theorems[begin_num:]

    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    return theorems

def evaluate(
    method: str, 
    data_path: str,
    jar_path: str,
    isabelle_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    begin_num: Optional[int] = None,
    runed_logs: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    use_sampling: bool = False,
    history_size: int = 1,
    timeout: int = 600,
    num_cpus: int = 1,
    with_gpus: bool = False,
    verbose: bool = False,
) -> float:
    set_logger(verbose)

    theorems = _get_theorems(
        "isabelle", data_path, split, file_path, full_name, name_filter, num_theorems, begin_num, runed_logs,
    )
   
    repo = {
        "jar_path": jar_path,
        "isa_path": isabelle_path,
    }

    # # Search for proofs using multiple concurrent provers.
    if method == "step":

        repo = {
            "jar_path": jar_path,
            "isa_path": isabelle_path,
        }
        prover = DistributedProver(
            ckpt_path,
            indexed_corpus_path,
            tactic,
            module,
            num_cpus,
            with_gpus=with_gpus,
            timeout=timeout,
            num_sampled_tactics=num_sampled_tactics,
            use_sampling=use_sampling,
            history_size=history_size,
            debug=verbose,
        )
        results = prover.search_unordered(repo, theorems)
    elif method == "cheating":
        database = {}
        for root, _, files in os.walk(data_path):
            for file in files:
                full_dir = os.path.join(root, file)
                with open(full_dir, "r", encoding="utf8") as f:
                    data = json.load(f)
                    for d in data:
                        lemma = d['full_name']
                        proof = d['full_proof_text'].split(lemma)[-1].strip()
                        database[lemma] = proof
        prover = CheatingProver(database=database, timeout=300, debug=True)
        results = prover.search(repo, theorems[0])
    else:
        raise NotImplementedError

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    pickle_path = f"{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["step", "cheating"],
        default="cheating",
        help="The formal system to use for evaluation.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/e/data/FVELer_benchmark_v0.1/",
        help="Path to the data extracted by LeanDojo/IsaDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument(
        "--jar-path",
        type=str,
        help="Path to the jar file.",
        # default="multilevel_isabelle/target/scala-2.13/PISA-assembly-0.1.jar"
        default="/home/xiaohlim/Portal-to-ISAbelle/target/scala-2.13/PISA-assembly-0.1.jar",
    )
    parser.add_argument("--isabelle-path", type=str, help="Path to the Isabelle installation.", default="/home/xiaohlim/Isabelle2022/")
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str) #default="lemma is_invarI[intro?]")
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument("--begin-num", type=int)
    parser.add_argument("--runed-logs", nargs="+")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/mnt/e/model/Llama-3-8b-Instruct",
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=16,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--with-gpus", action="store_true", help="Use GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    args.with_gpus = True

    pass_1 = evaluate(
        args.method,
        args.data_path,
        args.jar_path,
        args.isabelle_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.begin_num,
        args.runed_logs,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.use_sampling,
        args.history_size,
        args.timeout,
        args.num_cpus,
        args.with_gpus,
        args.verbose,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
