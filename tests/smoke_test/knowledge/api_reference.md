# greet CLI Reference

Generate a JSON greeting for a name.

```bash
python greet.py NAME [--greeting WORD]
```

Parameters:
- `NAME` (positional, required) — Name to greet.
- `--greeting` (optional) — Greeting word. Default: `Hello`.

## Output

On success, prints a JSON object to stdout and exits 0:

```json
{
  "message": "Hello, DSAGT!",
  "status": "ok"
}
```

## Exit codes

- `0` — success.
- `1` — invalid input. The JSON output carries `"status": "error"` and an
  error `code` field (see troubleshooting).
