# Troubleshooting

## Out of memory errors

If you encounter out of memory errors when loading large files,
enable lazy loading by passing `lazy=True` to the DataLoader constructor.
This streams data in chunks rather than loading the entire file at once.

## Permission denied

Check that you have read access to the data directory.
On Linux, run: chmod -R 755 /path/to/data
