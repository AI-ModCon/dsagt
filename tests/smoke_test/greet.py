"""greet — the smoke test's registerable, executable fixture CLI.

Stdlib-only so registration + execution via dsagt-run needs no
dependency install.  The GRT-42 empty-name error code is documented in
knowledge/troubleshooting.md; the smoke script asks the agent to
retrieve it from the knowledge collection, so keep code and docs in
sync.
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Generate a greeting")
parser.add_argument("name", help="Name to greet")
parser.add_argument(
    "--greeting", default="Hello", help="Greeting word (default: Hello)"
)
args = parser.parse_args()

if not args.name.strip():
    print(json.dumps({"status": "error", "code": "GRT-42", "error": "empty name"}))
    sys.exit(1)

result = {"message": f"{args.greeting}, {args.name}!", "status": "ok"}
print(json.dumps(result, indent=2))
