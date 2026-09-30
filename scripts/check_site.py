"""Crawl a built MkDocs site: broken internal links/anchors, missing assets, placeholder text."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlparse

root = Path(sys.argv[1]).resolve()
BAD = re.compile(r"\b(placeholder|PENDING|TODO|coming with|lorem ipsum|TBD)\b", re.I)


class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.text = [], set(), []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if k in a and a[k]:
                self.links.append((tag, a[k]))
        if tag in ("script", "style", "code", "pre"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "code", "pre") and self._skip:
            self._skip -= 1

    def handle_data(self, d):
        if not self._skip:
            self.text.append(d)


pages = {}
for f in root.rglob("*.html"):
    p = P()
    p.feed(f.read_text(errors="ignore"))
    pages[f] = p
problems = []
for f, p in pages.items():
    rel = f.relative_to(root)
    for m in BAD.finditer(" ".join(p.text)):
        ctx = " ".join(p.text)[max(0, m.start() - 60): m.end() + 40].replace("\n", " ")
        problems.append(f"TEXT  {rel}: ...{ctx}...")
    for tag, link in p.links:
        u = urlparse(link)
        if u.scheme or link.startswith(("mailto:", "data:", "#")) and not link.startswith("#"):
            continue
        path, frag = urldefrag(link)
        if not path:
            target = f
        else:
            target = (f.parent / unquote(path)).resolve()
            if target.is_dir():
                target = target / "index.html"
        if not target.exists():
            problems.append(f"LINK  {rel}: {link}")
            continue
        if frag and target.suffix == ".html" and target in pages and frag not in pages[target].ids:
            problems.append(f"ANCHOR {rel}: {link}")
for x in sorted(set(problems)):
    print(x)
print(f"{len(pages)} pages, {len(set(problems))} problems")
