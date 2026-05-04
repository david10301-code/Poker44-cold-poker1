from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from poker44.miner.config import repository_paths


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = repository_paths(REPO_ROOT)
UPSTREAM_ROOT_CANDIDATES = (
    REPO_ROOT.parent / "Poker44_v1" / "Poker44-subnet",
    REPO_ROOT.parent / "Poker44-subnet",
    REPO_ROOT.parent / "Poker44-main",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic bot hands using the upstream Poker44 generator.")
    parser.add_argument("--output", type=str, default=str(PATHS.generated_bot_hands_path))
    parser.add_argument("--num-hands-to-play", type=int, default=40000)
    parser.add_argument("--num-hands-to-select", type=int, default=32000)
    parser.add_argument("--hands-per-session", type=int, default=50)
    parser.add_argument("--seed", type=int, default=424242)
    return parser.parse_args()


def _load_upstream_module():
    upstream_root = None
    upstream_generator = None
    for candidate_root in UPSTREAM_ROOT_CANDIDATES:
        candidate_generator = (
            candidate_root / "hands_generator" / "bot_hands" / "generate_poker_data.py"
        )
        if candidate_generator.exists():
            upstream_root = candidate_root
            upstream_generator = candidate_generator
            break

    if upstream_root is None or upstream_generator is None:
        searched = ", ".join(str(path) for path in UPSTREAM_ROOT_CANDIDATES)
        raise FileNotFoundError(
            "Could not find an upstream bot-hand generator. "
            f"Searched roots: {searched}"
        )

    spec = importlib.util.spec_from_file_location(
        "poker44_upstream_bot_generator",
        upstream_generator,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load upstream generator module from {upstream_generator}"
        )
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = _load_upstream_module()

    generator = module.PokerHandGenerator(seed=args.seed)
    profiles = [
        module.BotProfile(name="tight_aggressive", tightness=0.70, aggression=0.75, bluff_freq=0.05),
        module.BotProfile(name="loose_aggressive", tightness=0.40, aggression=0.80, bluff_freq=0.12),
        module.BotProfile(name="tight_passive", tightness=0.68, aggression=0.35, bluff_freq=0.03),
        module.BotProfile(name="loose_passive", tightness=0.42, aggression=0.30, bluff_freq=0.08),
        module.BotProfile(name="balanced", tightness=0.55, aggression=0.55, bluff_freq=0.08),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hands = generator.generate_hands(
        num_hands_to_play=args.num_hands_to_play,
        num_hands_to_select=args.num_hands_to_select,
        bot_profiles=profiles,
        output_file=str(output_path),
        hands_per_session=args.hands_per_session,
    )

    if not output_path.exists():
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(hands, handle)

    print(f"Generated {len(hands)} bot hands at {output_path}")


if __name__ == "__main__":
    main()
