// Size documentation figures by their shape so every figure keeps good proportions:
// square plots stay compact, panoramic strips use the full column.
// Figures that already carry a shape class (sk-fig--square / plot / wide / strip) are sized by CSS alone;
// this script classifies the rest by the image's natural aspect ratio (data-shape).
(function () {
  function classify(img) {
    var w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return;
    var ar = w / h;
    img.setAttribute("data-shape", ar < 1.15 ? "square" : ar < 1.9 ? "plot" : ar < 3.3 ? "wide" : "strip");
  }
  function run() {
    document.querySelectorAll(".md-typeset figure img, .md-typeset p > img").forEach(function (img) {
      if (img.hasAttribute("data-shape")) return;
      if (img.closest(".sk-section, .sk-hero, .grid, .sek-meta, .sk-curves, [class*='sk-fig--']")) return;
      if (/\.svg(\?|#|$)/.test(img.getAttribute("src") || "")) return;
      if (img.complete && img.naturalWidth) classify(img);
      else img.addEventListener("load", function () { classify(img); }, { once: true });
    });
  }
  // Material's instant navigation exposes document$; subscribe once it exists, else run on load.
  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(run);
  else if (document.readyState !== "loading") run();
  else document.addEventListener("DOMContentLoaded", run);
})();
