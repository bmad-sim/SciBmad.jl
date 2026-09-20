// Keep the reader's place in the left sidebar across page loads.
//
// Furo renders the sidebar fresh on every page, so its scroll box always starts
// at the top: click an entry halfway down a long menu and the next page shows
// the menu from the beginning, with the page just opened somewhere below the
// fold. Scroll the entry for the current page up to the top of the menu instead,
// so it -- and the sections of it that Furo expands underneath -- are what the
// reader sees.
(function () {
  // A little room above the entry, so it does not sit flush against the edge
  // and the reader can tell that the menu is scrolled.
  var MARGIN = 12;

  function alignCurrentEntry() {
    var box = document.querySelector(".sidebar-scroll");
    if (!box) return;
    // Furo marks the entry for the page being shown with `current-page`; take
    // its own link, not the `current` links of the ancestors containing it.
    var entry = box.querySelector(".current-page > a.current");
    if (!entry) return;  // a page outside the toctree, e.g. the search results
    var item = entry.closest("li");  // the entry together with its sections

    var boxTop = box.getBoundingClientRect().top;
    var itemRect = item.getBoundingClientRect();
    // Nothing to do when the entry and all of its sections are already on
    // screen -- which is the case for the first entries of the menu, and they
    // should stay where they are rather than being nudged by the margin.
    if (itemRect.top >= boxTop && itemRect.bottom <= boxTop + box.clientHeight) {
      return;
    }

    // Work from the on-screen distance between entry and box, so the result
    // does not depend on where the sidebar itself sits on the page. Scrolling
    // past either end of the range is not possible, so an entry at the very
    // bottom of the menu simply comes as close to the top as it can.
    var target = box.scrollTop + entry.getBoundingClientRect().top - boxTop - MARGIN;

    // `.sidebar-scroll` is `scroll-behavior: smooth`, which is right for a
    // click but not here -- the menu should be in place when the page appears,
    // not animate into it.
    var behavior = box.style.scrollBehavior;
    box.style.scrollBehavior = "auto";
    box.scrollTop = target;
    box.style.scrollBehavior = behavior;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", alignCurrentEntry);
  } else {
    alignCurrentEntry();
  }
})();
