# greet Troubleshooting

## Error GRT-42: empty name

When the `NAME` argument is empty or whitespace-only, greet prints a JSON
error object with error code `GRT-42` and exits 1:

```json
{"status": "error", "code": "GRT-42", "error": "empty name"}
```

Fix: pass a non-empty name. Shell quoting is the usual culprit — an unset
variable like `"$USER_NAME"` expands to an empty string.

## Output is not valid JSON

greet writes only JSON to stdout. If you see extra text mixed in, another
tool in your pipeline is writing to the same stream — redirect its output
to stderr.

## Wrong greeting word

`--greeting` must come after the name or be joined with `=`
(`--greeting=Ahoy`); a bare `--greeting` at the end consumes the name as
its value and fails with a missing-argument error.
