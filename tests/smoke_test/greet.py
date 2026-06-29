"""Simple greeting script for smoke testing."""

import argparse
import json

parser = argparse.ArgumentParser(description="Generate a greeting")
parser.add_argument("name", help="Name to greet")
parser.add_argument(
    "--greeting", default="Hello", help="Greeting word (default: Hello)"
)
args = parser.parse_args()

result = {"message": f"{args.greeting}, {args.name}!", "status": "ok"}
print(json.dumps(result, indent=2))
