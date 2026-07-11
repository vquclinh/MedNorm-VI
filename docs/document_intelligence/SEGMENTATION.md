# L1 Segmentation

All segmentation preserves exact offsets and delimiters. Only **lines +
newline delimiters** are required to tile the whole document; sentences, list
items, rows, and tokens may be sparse.

## Lines & paragraphs

`lines.split_lines` splits on `\n` and `\r\n`, keeping each terminator as an
explicit **delimiter** node. A line's content span **excludes** its terminator;
content lines plus newline delimiters cover `[0, len)` with no gaps or overlaps.
Blank lines are kept as zero-length content lines; a trailing newline implies a
final empty line. Paragraphs are maximal runs of non-blank lines, and never
cross a section start (`paragraphs(..., break_before=header_lines)`).

Preserved: LF, CRLF, blank-line boundaries, multiple blank lines, indented lines,
empty documents, and documents with no trailing newline.

## Sentence-like segmentation (conservative)

`sentences.segment_sentences` does **not** assume every period ends a sentence:

- a period **between digits** is a decimal, not a boundary (`38.5`);
- a period after a known **abbreviation** or a single-letter **initial** is not a
  boundary (`500 mg. po`, `A.`);
- `;` splits (semicolon-heavy clinical text), `\n` separates sentence-like spans.

Sentence spans keep their terminating punctuation; the whitespace between
sentences belongs to no sentence.

## Numbered & bulleted lists

`lists.detect_list_item` recognizes `1.`, `1)`, `(1)`, `a.`, `a)`, `-`, `–`, `—`,
`*`, `•`, and other bullet variants. It stores the **marker span** and the
**content span** separately. **The list number/bullet is never folded into the
item content** — later entity spans must not swallow it.

## Table-like & key-value rows

`table_rows.detect_row` labels rows structurally only:

- tab-separated → `table_like` (tab-split cells);
- `name: value` → `key_value_like` (key cell + value cell);
- `;`-heavy rows → `table_like`.

**Non-exclusive representation.** A colon line is simultaneously a
`key_value_like` row (with exact `key`/`value` cell offsets) **and** carries
sentence-like segment(s) over the narrative value (nested inside the value cell).
So `Chẩn đoán: Viêm phổi cộng đồng.` yields a row, a key cell, a value cell, and a
value sentence — structural views are not forced to be mutually exclusive. Tokens
reference the **smallest** covering segment (the value sentence over the row).

L1 emits **no** `TÊN_XÉT_NGHIỆM`/`KẾT_QUẢ_XÉT_NGHIỆM` (or any) entities — that is a
later layer's job.

## Token view (canonical, not a model tokenizer)

`tokens.tokenize` produces word/number tokens that keep internal decimal
separators, unit slashes, and hyphens (`14.43`, `14,43`, `325-650`, `mg/ml`),
plus one token per punctuation mark. Each token stores absolute offsets, a
normalized form, category (`word`/`number`/`punct`), punctuation flag, and
whitespace-before. Model-specific tokenizers added later must map back through
this L1 token view.
