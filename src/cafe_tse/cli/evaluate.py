from __future__ import annotations

import argparse

from cafe_tse.engine.evaluator import Evaluator
from cafe_tse.utils.config import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_audio", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    summary = Evaluator(cfg, args.checkpoint, args.test_manifest, args.out_dir, args.device).run(args.save_audio)
    print(summary)


if __name__ == "__main__":
    main()

