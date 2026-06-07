from __future__ import annotations

import argparse

from cafe_tse.engine.trainer import Trainer
from cafe_tse.utils.config import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--valid_manifest", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    result = Trainer(cfg, args.train_manifest, args.valid_manifest, args.exp_dir).fit()
    print(result)


if __name__ == "__main__":
    main()

