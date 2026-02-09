"""
Reflow paragraphs in a text file to a fixed line width (default 80 chars).
Preserves blank lines and numbered paragraph prefixes like "1. ".

Usage:
    python3 scripts/reflow_paragraphs.py path/to/file.txt --width 80

The script overwrites the file with the reflowed version. It is safe for plain text files.
"""
import argparse
import textwrap
from pathlib import Path


def reflow_text(text, width=80, max_words_per_line=None):
    # Split into paragraphs by blank lines
    paras = []
    cur = []
    for line in text.splitlines():
        if line.strip() == '':
            if cur:
                paras.append(' '.join(l.strip() for l in cur))
                cur = []
            else:
                paras.append('')
        else:
            cur.append(line)
    if cur:
        paras.append(' '.join(l.strip() for l in cur))

    out_lines = []
    for p in paras:
        if p == '':
            out_lines.append('')
            continue
        # Try to preserve numbered prefix like "1. " or "12. " or headings "Part 1:"
        prefix = ''
        rest = p
        m = None
        import re
        m = re.match(r'^(\s*\d{1,3}\.\s+)(.*)$', p)
        if m:
            prefix = m.group(1)
            rest = m.group(2)
        # Also keep short headings (up to 4 words) intact as a single line
        if len(rest.split()) <= 10 and rest.isupper():
            out_lines.append(prefix + rest)
            continue
        if max_words_per_line and max_words_per_line > 0:
            # wrap by words per line
            words = rest.split()
            line = []
            lines = []
            for w in words:
                line.append(w)
                if len(line) >= max_words_per_line:
                    lines.append(' '.join(line))
                    line = []
            if line:
                lines.append(' '.join(line))
            # reattach prefix to first line
            if prefix and lines:
                lines[0] = prefix + lines[0]
            out_lines.extend(lines)
            out_lines.append('')
        else:
            wrapped = textwrap.fill(rest, width=width)
            # reattach prefix to the first line
            if prefix:
                wrapped_lines = wrapped.splitlines()
                wrapped_lines[0] = prefix + wrapped_lines[0]
                out_lines.extend(wrapped_lines)
            else:
                out_lines.extend(wrapped.splitlines())
            out_lines.append('')
    # Remove trailing blank line
    if out_lines and out_lines[-1] == '':
        out_lines = out_lines[:-1]
    return '\n'.join(out_lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to reflow')
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--max-words', type=int, default=None, help='Maximum words per line (overrides --width when set)')
    args = parser.parse_args()
    p = Path(args.file)
    if not p.exists():
        print('File not found:', p)
        return
    text = p.read_text(encoding='utf-8')
    new = reflow_text(text, width=args.width, max_words_per_line=args.max_words)
    p.write_text(new, encoding='utf-8')
    if args.max_words:
        print(f'Reflowed {p} -> max_words_per_line={args.max_words}')
    else:
        print(f'Reflowed {p} -> width={args.width}')

if __name__ == '__main__':
    main()
