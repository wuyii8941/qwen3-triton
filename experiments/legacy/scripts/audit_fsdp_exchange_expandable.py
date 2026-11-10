"""Fresh-process CLI for the matched expandable full quality campaign."""
import argparse
import json
from pathlib import Path
from scripts.campaign_fsdp_exchange_expandable import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = audit(args.root, allow_partial=args.allow_partial)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
