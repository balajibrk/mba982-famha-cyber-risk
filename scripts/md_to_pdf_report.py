"""Convert docs/MBA982_Project_Module_Report.md to a print-ready PDF
(same styling approach as the MBA979 module report)."""
from __future__ import annotations

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "MBA982_Project_Module_Report.md"
PDF_PATH = ROOT / "docs" / "MBA982_Project_Module_Report.pdf"

CSS = """
@page {
  size: A4;
  margin: 1.8cm 1.6cm 2.0cm 1.6cm;
  @frame footer {
    -pdf-frame-content: footerContent;
    bottom: 0.6cm;
    margin-left: 1.6cm;
    margin-right: 1.6cm;
    height: 1.0cm;
  }
}
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1a1a1a;
}
h1 {
  font-size: 16pt;
  color: #0f172a;
  border-bottom: 2px solid #1e3a5f;
  padding-bottom: 6px;
  margin-top: 0;
  margin-bottom: 12px;
}
h2 {
  font-size: 12.5pt;
  color: #1e3a5f;
  margin-top: 18px;
  margin-bottom: 8px;
  border-bottom: 1px solid #cbd5e1;
  padding-bottom: 3px;
}
h3 {
  font-size: 11pt;
  color: #334155;
  margin-top: 12px;
  margin-bottom: 6px;
}
p { margin: 0 0 8px 0; }
strong { color: #0f172a; }
a { color: #1d4ed8; text-decoration: none; }
blockquote {
  margin: 10px 0;
  padding: 8px 12px;
  background: #f1f5f9;
  border-left: 3px solid #1e3a5f;
  font-size: 10pt;
  color: #334155;
}
code {
  font-family: Courier, monospace;
  font-size: 8.5pt;
  background: #f8fafc;
}
pre {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 8px 10px;
  font-size: 8pt;
  line-height: 1.35;
  white-space: pre-wrap;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0 12px 0;
  font-size: 9pt;
}
th {
  background: #1e3a5f;
  color: #ffffff;
  text-align: left;
  padding: 5px 6px;
  font-weight: bold;
}
td {
  border: 1px solid #cbd5e1;
  padding: 4px 6px;
  vertical-align: top;
}
tr:nth-child(even) td { background: #f8fafc; }
ul, ol { margin: 4px 0 10px 18px; padding: 0; }
li { margin-bottom: 3px; }
.footer {
  font-size: 8pt;
  color: #64748b;
  text-align: center;
  border-top: 1px solid #e2e8f0;
  padding-top: 4px;
}
"""


def md_to_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>MBA982 Project Module Report</title>
<style>{CSS}</style>
</head>
<body>
{body}
<div id="footerContent" class="footer">
  MBA982 Project Module Report - Confidential for faculty review
  | https://github.com/balajibrk/mba982-famha-cyber-risk
  | Page <pdf:pagenumber/> of <pdf:pagecount/>
</div>
</body>
</html>
"""


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html = md_to_html(md_text)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PDF_PATH.open("wb") as out:
        status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if status.err:
        raise SystemExit(f"PDF conversion failed with {status.err} error(s)")
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
