// The Julia / Python switch in the top bar. The site is built twice (docs/_ext/dual_language.py):
// the Julia pages at the root of the site, the Python pages under python/. The switch goes to the
// same page (and section) in the other language and remembers the choice, so that pages opened
// later (e.g. from a search engine) come in that language too.
(function () {
  var KEY = "scibmad-docs-language";
  var lang = window.SCIBMAD_DOCS_LANGUAGE || "julia";
  var rootAttr = document.documentElement.getAttribute("data-content_root") || "./";
  var root = new URL(rootAttr, window.location.href).href; // this language's root
  var here = window.location.href.split("#")[0]; // (the section is added when switching)
  var rel = here.indexOf(root) === 0 ? here.slice(root.length) : "";

  function urlFor(target) {
    if (target === lang) return here;
    var base = lang === "julia" ? root + "python/" : new URL("../", root).href;
    return base + rel;
  }

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(v) {
    try { window.localStorage.setItem(KEY, v); } catch (e) { /* private mode */ }
  }

  // A page opened in the other language than the one chosen: go to the chosen one
  // (`?lang=julia` or `?lang=python` in the address overrides and stores the choice).
  var param = new URLSearchParams(window.location.search).get("lang");
  if (param === "julia" || param === "python") store(param);
  var pref = stored();
  if (pref && pref !== lang && window.location.protocol !== "file:") {
    window.location.replace(urlFor(pref).replace(/[?&]lang=(julia|python)/, "") +
                            window.location.hash);
    return;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var container = document.querySelector(".content-icon-container");
    if (!container) return;
    var box = document.createElement("div");
    box.className = "language-switch";
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", "Language of the code examples");
    ["julia", "python"].forEach(function (l) {
      var a = document.createElement("a");
      a.textContent = l === "julia" ? "Julia" : "Python";
      a.href = urlFor(l);
      a.className = "language-switch-option" + (l === lang ? " active" : "");
      if (l === lang) a.setAttribute("aria-current", "true");
      a.title = l === lang ? "The code examples are in " + a.textContent
                           : "Show the code examples in " + a.textContent;
      a.addEventListener("click", function (ev) {
        store(l);
        if (l === lang) { ev.preventDefault(); return; }
        a.href = urlFor(l) + window.location.hash;
      });
      box.appendChild(a);
    });
    container.insertBefore(box, container.firstChild);
  });
})();
