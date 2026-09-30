// MathJax 3 configuration for pymdownx.arithmatex (generic mode).
// Arithmatex wraps every formula, whatever delimiter the source used ($...$, $$...$$, \(...\), \[...\]),
// in <span|div class="arithmatex"> with \( \) or \[ \] delimiters, so MathJax only has to look inside
// those elements. Everything else on the page (code, tables of file paths, prices ...) is left alone.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "none"
  },
  chtml: {
    // match the surrounding serif text size; keep the default MathJax TeX font for glyph coverage
    matchFontHeight: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// With navigation.instant, Material swaps page content without a full reload and emits document$.
// Re-typeset on every such swap. The guard covers the first emission, which can arrive before the
// MathJax bundle has finished loading (MathJax then typesets the initial page itself on startup).
document$.subscribe(() => {
  if (!window.MathJax || !MathJax.startup || !MathJax.typesetPromise) return;
  MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
