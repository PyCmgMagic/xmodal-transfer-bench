#!/usr/bin/env python3
"""Entity MCQ evaluation with brief CoT and no concrete boxed-letter examples."""
import argparse
from pathlib import Path

from answer_readout_entity_v4 import parse_answer, VERSION
from entityqa_common import save, now
from entityqa_prompt_neutral_v5 import eval_payload, COT_SUFFIX, PROMPT_VERSION
from evaluate_entityqa_fullfact_v2 import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'output', 'model']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arm', required=True, choices=['base', 'TEXT', 'VISION'])
    p.add_argument('--gpu', default='0')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-tokens', type=int, default=65536)
    p.add_argument('--wall-seconds', type=int, default=180)
    p.add_argument('--import-predictions', type=Path,
                   help='Reuse base outputs only after the shared evaluator verifies every input hash and label')
    args = p.parse_args()
    try:
        run(args, payload_builder=eval_payload, cot_suffix=COT_SUFFIX,
            answer_parser=parse_answer, readout_version=VERSION,
            prompt_protocol=PROMPT_VERSION + ': brief prompted CoT; user-only neutral instruction; no concrete letter examples',
            evaluator_version=5)
    except Exception as error:
        save(args.output / 'failure.json', dict(error=repr(error), time=now()))
        raise


if __name__ == '__main__':
    main()
