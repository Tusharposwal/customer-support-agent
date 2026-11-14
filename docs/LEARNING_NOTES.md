# Learning Notes

## Knowledge-base loading and validation

### How the knowledge base is loaded

The support articles live in `data/kb.json` as a JSON list. Each article looks like this:

```json
{"id": "KB-001", "title": "Reset your password", "text": "...", "url": "https://..."}
```

`load_articles()` in `src/support_agent/kb.py` opens that file, parses it with
`json.load`, validates it, and returns a list of Python dictionaries. Keyword
retrieval (`retrieve()`) and semantic retrieval (`semantic_kb.py`) both call it.

### Why validation is necessary

Everything downstream trusts the article data. Without checks, a typo in
`kb.json` (a missing `url`, an empty `text`, a repeated `id`) would not fail
when the file is loaded. It would surface later as a confusing `KeyError` deep
in retrieval, or worse, as an answer citing a blank or wrong source. Validation
makes bad data fail **early, loudly, and with a message that says where the
problem is**.

### How the validation works

`load_articles()` raises a `ValueError` with a clear message when:

| Check | Example message |
| --- | --- |
| File is not valid JSON | `Knowledge base ... is not valid JSON: Expecting value (line 3, column 5)` |
| Top level is not a list | `... must contain a JSON list of articles, got dict` |
| An article is not an object | `..., article #1 must be a JSON object, got str` |
| A required field is missing | `..., article #0 is missing required field(s): url` |
| A field is not a string | `..., article #0 field 'id' must be a string, got int` |
| A field is empty or only spaces | `..., article #0 field 'text' must not be empty` |
| Two articles share an id | `..., article #2 has duplicate id 'KB-001'` |

Required fields are listed once in `REQUIRED_FIELDS = ("id", "title", "text", "url")`.
Extra fields are allowed, so new fields can be added to articles later without
changing the validator.

### Files that implement it

- `src/support_agent/kb.py`: `load_articles()` and `_validate_articles()`.
- `tests/test_kb.py`: offline tests that write small temporary KB files with
  pytest's `tmp_path` fixture and check that each bad case raises the right error.

### Technical decisions (in plain language)

- **Plain Python checks instead of a library (such as Pydantic).** There are
  only four string fields. A short function is easier to read and adds no
  dependency. Pydantic can come later for the LLM's structured output, where
  it is more valuable.
- **One error type (`ValueError`).** Callers only need to catch one kind of
  error, and the message explains the details.
- **Invalid JSON is re-raised as `ValueError` using `from error`.** The message
  includes the file path and line/column, and the original error stays
  attached for debugging.
- **Optional `path` argument.** `load_articles()` still reads `data/kb.json` by
  default, so existing code is unchanged. Tests pass a temporary file, so they
  never touch the real data and never call an external API.
- **Stop at the first problem.** The validator stops at the first error it
  finds. That keeps the code simple, and the knowledge base is small.
- **Article numbers start at 0** (`article #0` is the first item), matching
  Python list indexes.

### Known issue (not caused by validation)

`test_email_change_is_not_supported` fails on the committed baseline as well.
"How do I change my email address?" matches KB-001 because the password article
mentions "email address" (score 0.667 ≥ `MIN_SCORE` 0.30). Fixing this is a
retrieval change and is tracked separately.
