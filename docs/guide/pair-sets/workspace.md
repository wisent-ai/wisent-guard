# Pair sets in the workspace

`ster workspace import-pairs <file> [--name <id>]` (and
`ster onboarding --import-pairs <file>`) takes a canonical pair-set document
into the workspace under `$XDG_DATA_HOME/ster` (or `~/.local/share/ster`) and
makes it the active set. The whole document is validated before anything is
written, and a rejected import leaves no partial state.

## The set's id

- With `--name`, the id is that name. It must start with an ASCII letter or
  digit and hold only letters, digits, dots, underscores and hyphens.
- Without `--name`, the id is the file name's letters and digits, lowercased,
  with each run of other characters becoming one hyphen: `My Pairs.json`
  becomes `my-pairs`.
- A file name that holds no letter or digit, or is not UTF-8, gives no id. The
  import is `rejected`, naming the file and asking for `--name`; no id is made
  up for it.

## Recognising a set again

A set is identified by its content: the BLAKE2b-512 digest of its canonical
document, of which the first 16 bytes are kept, written as 32 hex characters.
This is the `digest` each entry of the workspace state records.

| Report status | When |
|---|---|
| `imported` | New content under a new id |
| `unchanged` | The same content was imported before, under any id; that set becomes active |
| `conflicting` | The id is taken by different content, or the destination file exists outside the workspace index; nothing is overwritten |
| `rejected` | The document is invalid, the `--name` is not a valid id, or the file name gives no id |
