/* swarmint site interactions — shared by index.html and docs.html.
   The hero swarm animation lives inline in index.html (page-specific). */
(function () {
  "use strict";

  /* ---- theme toggle: Auto (follow OS) -> Light -> Dark, persisted ---- */
  (function () {
    var KEY = "swarmint-theme", MODES = ["auto", "light", "dark"];
    var LABEL = { auto: "◐ Auto", light: "☀ Light", dark: "☾ Dark" };
    var root = document.documentElement, btn = document.getElementById("theme-toggle");
    var cur;
    try { cur = localStorage.getItem(KEY) || "auto"; } catch (e) { cur = "auto"; }
    if (MODES.indexOf(cur) < 0) cur = "auto";
    function apply(m) {
      if (m === "auto") root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", m);
      if (btn) btn.textContent = LABEL[m];
      if (window.__swarmAccentRefresh) window.__swarmAccentRefresh(); // recolor the swarm
    }
    apply(cur);
    if (btn) btn.addEventListener("click", function () {
      cur = MODES[(MODES.indexOf(cur) + 1) % MODES.length];
      try { localStorage.setItem(KEY, cur); } catch (e) {}
      apply(cur);
    });
  })();

  /* ---- copy-to-clipboard on code blocks ---- */
  (function () {
    var blocks = document.querySelectorAll(".codeblock");
    Array.prototype.forEach.call(blocks, function (block) {
      var pre = block.querySelector("pre");
      if (!pre) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copybtn";
      btn.setAttribute("aria-label", "Copy code to clipboard");
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        // Copy visible text minus the button label; comments (.c) are kept.
        var text = pre.innerText.replace(/\n?Copy(ied!)?$/, "");
        var done = function () {
          btn.textContent = "Copied!"; btn.classList.add("copied");
          setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, done);
        } else {
          var ta = document.createElement("textarea");
          ta.value = text; document.body.appendChild(ta); ta.select();
          try { document.execCommand("copy"); } catch (e) {}
          document.body.removeChild(ta); done();
        }
      });
      block.appendChild(btn);
    });
  })();

  /* ---- scroll reveal (respects reduced-motion via CSS fallback) ---- */
  (function () {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (!("IntersectionObserver" in window)) {
      Array.prototype.forEach.call(els, function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    Array.prototype.forEach.call(els, function (el) { io.observe(el); });
  })();

  /* ---- docs TOC: highlight the section in view ---- */
  (function () {
    var links = document.querySelectorAll(".doc-toc a");
    if (!links.length || !("IntersectionObserver" in window)) return;
    var map = {};
    Array.prototype.forEach.call(links, function (a) {
      var id = a.getAttribute("href").replace("#", "");
      if (id) map[id] = a;
    });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          Array.prototype.forEach.call(links, function (a) { a.classList.remove("active"); });
          if (map[e.target.id]) map[e.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    Object.keys(map).forEach(function (id) {
      var sec = document.getElementById(id);
      if (sec) io.observe(sec);
    });
  })();
})();
