"""Convert markdown solution papers to PDF using markdown + weasyprint."""

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 1in; }
body { font-family: 'Georgia', serif; font-size: 11pt; line-height: 1.5; color: #222; }
h1 { font-size: 16pt; margin-top: 1.2em; margin-bottom: 0.4em; color: #1a1a1a; }
h2 { font-size: 13pt; margin-top: 1em; margin-bottom: 0.3em; color: #333; }
h3 { font-size: 11pt; margin-top: 0.8em; margin-bottom: 0.2em; }
p { margin: 0.4em 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 10pt; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; }
th { background: #f0f0f0; font-weight: bold; }
pre, code { font-family: 'Courier New', monospace; font-size: 9pt; background: #f5f5f5; }
pre { padding: 8px; border: 1px solid #ddd; overflow-x: auto; white-space: pre-wrap; }
blockquote { border-left: 3px solid #ccc; margin: 0.5em 0; padding: 0.3em 1em;
             font-style: italic; color: #555; }
em { font-style: italic; }
strong { font-weight: bold; }
.title { font-size: 18pt; text-align: center; margin-bottom: 0.2em; }
.subtitle { font-size: 12pt; text-align: center; color: #666; margin-bottom: 0.1em; }
.author { font-size: 11pt; text-align: center; color: #666; margin-bottom: 1.5em; }
"""


def convert(md_path: str) -> None:
    src = Path(md_path)
    text = src.read_text(encoding="utf-8")

    # Extract YAML front matter for title/subtitle/author
    title = subtitle = author = ""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            front = text[3:end]
            text = text[end + 3 :].strip()
            for line in front.strip().split("\n"):
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("subtitle:"):
                    subtitle = line.split(":", 1)[1].strip().strip('"')
                elif line.startswith("author:"):
                    author = line.split(":", 1)[1].strip().strip('"')

    header_html = ""
    if title:
        header_html += f'<div class="title">{title}</div>'
    if subtitle:
        header_html += f'<div class="subtitle">{subtitle}</div>'
    if author:
        header_html += f'<div class="author">{author}</div>'

    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = f"<html><head><style>{CSS}</style></head><body>{header_html}{body}</body></html>"

    scratch = src.parent.parent / "scratch"
    scratch.mkdir(exist_ok=True)
    out = scratch / src.with_suffix(".pdf").name
    HTML(string=html).write_pdf(str(out))
    print(f"  {src} → {out}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        convert(path)
