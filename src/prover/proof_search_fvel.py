from math import exp
import sys
import time
import heapq
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from pisa.src.main.python.pisa_client import (DojoCrashError, DojoHardTimeoutError,
                                                             DojoInitError, TacticState, ProofFinished, IsabelleError, ProofGivenUp,
                                                             IsaDojo, Theorem)
from prover.search_tree import InternalNode, Edge, ProofFinishedNode, ErrorNode


class Status(Enum):
    """Status of a node or a proof search."""
    HALF_PROVED = "HalfProved"   # This node is half proved, with complete verifiable sketeches.
    PROVED = "Proved"  # This node (or search) has at least one known proof.
    FAILED = "Failed"  # This node (or search) has exhausted its options and cannot be proved within the current run.
    OPEN = "Open"  # This node (or search) has not been proven or given up on yet.

@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: list[str] | None

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int

class CheatingProver:
    def __init__(
        self,
        database: dict[str, str],
        timeout: int,
        debug: bool,
    ) -> None:

        self.timeout = timeout
        self.database = database
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")

    def search(
        self,
        repo: dict,
        thm: Theorem,
    ) -> SearchResult | None:
        logger.info(f"Proving {thm}")

        self.theorem = thm
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        try:
            init_port = 8005
            with IsaDojo(
                port=init_port,
                jar_path=repo["jar_path"],
                isa_path=repo["isa_path"],
                working_directory=str(thm.working_directory),
                theory_file_path=str(thm.file_path),
                theorem_name=str(thm.full_name)
            ) as (dojo, init_state):
                if init_state:
                    self.dojo = dojo
                    self.root = InternalNode(
                        state=init_state,
                        cumulative_logprob=0.0
                    )
                    self.nodes = {init_state: self.root}
                    self.priority_queue = [self.root]

                    try:
                        self._cheating_search()
                    except DojoCrashError:
                        logger.warning(f"Dojo crashed when proving {thm}")
                        pass
                else:
                    logger.warning(f"IsaDojo fail to init when proving {thm}")
                    self.root = InternalNode(
                        state=init_state,
                        cumulative_logprob=0.0,
                    )
                    self.nodes = {init_state: self.root}

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None
        
    def _cheating_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= \
                self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.from_tactic
        else:
            assert False, "Why this will happen?"
        suggestions = self._generate_tactics(ts)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = [
            self._run_tactic(search_node, tactic, logprob)
            for tactic, logprob in suggestions
        ]

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1



    def _generate_tactics(self, ts: str) -> list[tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)
        suggestions = self.database[ts]

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return [(suggestions, 1)]

    def _run_tactic(self,
                    node: InternalNode,
                    tactic: str,
                    logprob: float,
    ) -> Edge:
        def log_result(msg, replace_newline=True):
            if replace_newline:
                msg = msg.replace("\n", " ")
            logger.debug(msg)

        t0 = time.monotonic()
        log_result(f"Running tactic: {tactic}")
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            result_node = self.nodes[response]
            if isinstance(response, ProofFinished):
                log_result(f"Result: proof successed! - {str(response.message)}")
            elif isinstance(response, IsabelleError):
                log_result(f"Result: tactic failed! - {str(response)}")
            else:
                log_result(f"Result: duplicate result ! - {str(response.pp)}")
        except KeyError:
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
                log_result(f'Result: proof successed! - {str(response.message)}')
            elif type(response) in (
                IsabelleError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
                log_result(f'Result: tactic failed! - {str(response)}')
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )
                log_result(f'Result: tactic success! - {response.pp}')

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge