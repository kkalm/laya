"""Inspect a Laya checkpoint's architecture locally.

By default the model is exported with `torch.export` and served in Google's Model Explorer, which
groups the graph's operations into collapsible layers named after the modules in
`laya/common.py` (`DecisionModel`, its encoder, head, scorer and act head). `--summary` prints a
torchinfo layer table with output shapes and parameter counts instead.

Neither viewer is a dependency of the package; install the one you use:

    pip install ai-edge-model-explorer   # the graph (default)
    pip install torchinfo                # --summary

Both run on CPU in this process; the first run downloads the checkpoint from the Hub.
"""
import argparse
import sys

import torch

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items
from laya.train import to_internal

# The graph is traced from one concrete input, so its shapes are this example's: 26 tokens and
# three option markers on the English checkpoint. The architecture does not depend on it.
EXAMPLE_STATE = "what is the capital of France"
EXAMPLE_QUESTION = {"type": "choice", "instructions": "Which option correctly answers the question?",
                    "criteria": ["Paris", "Vienna", "London"]}


def example_inputs(agent: Agent):
    """`DecisionModel.forward`'s positional inputs for the example, laid out as `Agent` lays them out."""
    q = to_internal("example", EXAMPLE_QUESTION)
    ids, markers = build_sequence(agent.tok, EXAMPLE_STATE, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
    batch = collate_items([[{"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}]], agent.tok.pad_token_id)
    return (batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"],
            batch["qtype"])


def _require(module: str, package: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit("visualise_model: this needs %s, which laya does not install: pip install %s"
                 % (package, package))


def print_summary(model: torch.nn.Module, inputs, depth: int) -> None:
    torchinfo = _require("torchinfo", "torchinfo")
    stats = torchinfo.summary(model, input_data=list(inputs), depth=depth, verbose=0,
                              col_names=("output_size", "num_params"), row_settings=("var_names",))
    print(stats)


def explore(model: torch.nn.Module, inputs, name: str, host: str, port: int) -> None:
    model_explorer = _require("model_explorer", "ai-edge-model-explorer")
    with torch.no_grad():
        program = torch.export.export(model, inputs)
    # Blocks until interrupted; Model Explorer prints the URL and tries to open a browser.
    model_explorer.visualize_pytorch(name, exported_program=program, host=host, port=port)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a Laya checkpoint's architecture")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--summary", action="store_true",
                        help="Print a torchinfo layer table instead of starting Model Explorer")
    parser.add_argument("--depth", type=int, default=2,
                        help="How many module levels --summary expands (default 2)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Model Explorer host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Model Explorer port (default 8080)")
    args = parser.parse_args(argv)

    agent = Agent(args.model, compile=False, device="cpu")
    model = agent.model.eval()
    inputs = example_inputs(agent)
    if args.summary:
        print_summary(model, inputs, args.depth)
    else:
        explore(model, inputs, args.model.rstrip("/").split("/")[-1], args.host, args.port)


if __name__ == "__main__":
    main()
