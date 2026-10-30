// "Contents" at the right edge of the dashboard: the headings of the open tab, highlighted by scroll position.
//
// Streamlit draws only inside its own elements, so ui/app.py installs this script into the page itself
// (through a zero-height component) once per page load. It then follows reruns and tab switches on its own:
// a MutationObserver rebuilds the list from the visible tab's h2/h3 headings, and every scroll recolours it.
// The colour of each entry blends from the text colour to the theme's primary colour as its section crosses
// the reading line (a fifth of the way down the window), so the highlight glides from one entry to the next
// as you scroll instead of jumping. A thin marker on the rail moves with it.
(function () {
  if (window.__rlContents) { window.__rlContents.rebuild(); return; }
  var doc = document;
  var ACCENT = "__ACCENT__";   // replaced by ui/app.py with the theme's primary colour
  var MIN_WIDTH = 1250;        // px; narrower windows have no room beside the content

  var nav = doc.createElement("nav");
  nav.id = "rl-contents";
  nav.setAttribute("aria-label", "Contents of this tab");
  nav.innerHTML = '<div class="rl-contents-title">Contents</div>'
    + '<div class="rl-contents-body"><div class="rl-contents-marker"></div><ol></ol></div>';
  var style = doc.createElement("style");
  style.textContent = [
    "#rl-contents { position: fixed; right: 1.25rem; top: 5.5rem; width: 15rem; z-index: 90; display: none;",
    "  font-size: 0.84rem; line-height: 1.35; }",
    "@media (min-width: " + MIN_WIDTH + "px) {",
    "  #rl-contents.rl-show { display: block; }",
    "  body.rl-contents-on [data-testid='stMainBlockContainer'] { padding-right: 18rem !important; }",
    "}",
    "#rl-contents .rl-contents-title { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;",
    "  text-transform: uppercase; opacity: 0.55; margin: 0 0 0.4rem 0.9rem; }",
    "#rl-contents .rl-contents-body { position: relative; max-height: calc(100vh - 8rem); overflow-y: auto;",
    "  border-left: 2px solid rgba(128, 128, 128, 0.25); }",
    "#rl-contents ol { list-style: none; margin: 0; padding: 0; }",
    "#rl-contents li a { display: block; padding: 0.22rem 0.6rem 0.22rem 0.8rem; text-decoration: none;",
    "  cursor: pointer; overflow-wrap: anywhere; transition: color 0.15s linear, opacity 0.15s linear; }",
    "#rl-contents li a:hover { text-decoration: underline; }",
    "#rl-contents li.rl-h2 a { font-weight: 600; }",
    "#rl-contents li.rl-h3 a { padding-left: 1.4rem; }",
    "#rl-contents .rl-contents-marker { position: absolute; left: -2px; width: 2px; height: 0;",
    "  border-radius: 1px; background: " + ACCENT + "; transition: top 0.12s linear, height 0.12s linear; }"
  ].join("\n");
  doc.head.appendChild(style);
  doc.body.appendChild(nav);
  var body = nav.querySelector(".rl-contents-body");
  var list = nav.querySelector("ol");
  var marker = nav.querySelector(".rl-contents-marker");

  var entries = [];   // {el: heading element, a: its link, li: its list item}
  var signature = "";
  var textRGB = [128, 128, 128];
  var accentRGB = hexToRGB(ACCENT);

  function hexToRGB(h) {
    var m = /^#?([0-9a-f]{6})$/i.exec(String(h).trim());
    if (!m) return [42, 120, 214];
    var n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function cssRGB(s) {
    var m = /rgba?\(([^)]+)\)/.exec(s || "");
    return m ? m[1].split(",").slice(0, 3).map(Number) : [128, 128, 128];
  }
  function visible(el) {   // false inside hidden tab panels and collapsed expanders (laid out, but clipped to no height)
    if (el.offsetParent === null) return false;
    for (var p = el.parentElement; p; p = p.parentElement) {
      if (p.tagName === "DETAILS" && !p.open) return false;
      if (p.clientHeight === 0 && getComputedStyle(p).overflow !== "visible") return false;
    }
    return true;
  }
  function activePanel() {
    var panels = doc.querySelectorAll("[data-testid='stTabPanel'], [role='tabpanel']");
    for (var i = 0; i < panels.length; i++) {
      var p = panels[i];
      if (!p.hidden && p.getAttribute("aria-hidden") !== "true" && visible(p)) return p;
    }
    return null;
  }
  function headerBottom() {
    var h = doc.querySelector("[data-testid='stHeader']");
    return h ? h.getBoundingClientRect().bottom : 0;
  }
  function scrollerOf(el) {
    for (var p = el.parentElement; p; p = p.parentElement) {
      var o = getComputedStyle(p).overflowY;
      if ((o === "auto" || o === "scroll") && p.scrollHeight > p.clientHeight + 1) return p;
    }
    return doc.scrollingElement || doc.documentElement;
  }

  function rebuild() {
    var panel = activePanel();
    var heads = [];
    if (panel) {
      var all = panel.querySelectorAll("h2, h3");
      for (var i = 0; i < all.length; i++) if (visible(all[i])) heads.push(all[i]);
    }
    var sig = heads.map(function (h) { return h.tagName + ":" + h.textContent.trim(); }).join("|");
    var connected = entries.every(function (e) { return e.el.isConnected; });
    if (sig === signature && connected && heads.length === entries.length) { update(); return; }
    signature = sig;
    textRGB = cssRGB(getComputedStyle(doc.body).color);
    list.textContent = "";
    entries = heads.map(function (h) {
      var li = doc.createElement("li");
      li.className = "rl-" + h.tagName.toLowerCase();
      var a = doc.createElement("a");
      a.textContent = h.textContent.trim();
      a.href = "#" + (h.id || "");
      a.addEventListener("click", function (ev) {
        ev.preventDefault();
        h.style.scrollMarginTop = (headerBottom() + 16) + "px";   // keep it out from under the sticky header
        h.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      li.appendChild(a);
      list.appendChild(li);
      return { el: h, a: a, li: li };
    });
    var show = entries.length > 1;
    nav.classList.toggle("rl-show", show);
    doc.body.classList.toggle("rl-contents-on", show);
    update();
  }

  function update() {
    if (!entries.length) return;
    var top = headerBottom();
    var line = top + 0.2 * (window.innerHeight - top);   // the reading line
    var pos = entries.map(function (e) { return e.el.getBoundingClientRect().top - line; });
    var current = 0;
    for (var k = 0; k < pos.length; k++) if (pos[k] <= 0) current = k;
    // Continuous position: the current entry, plus the fraction of the way to the next heading. The entry stays
    // solid through the first half of its section, then crosses over to the next one as its heading approaches.
    var p = current;
    if (current + 1 < pos.length) {
      var span = pos[current + 1] - pos[current];
      if (span > 0) p = current + Math.max(0, 2 * Math.min(1, -pos[current] / span) - 1);
    }
    if (pos[0] > 0) p = 0;
    var sc = scrollerOf(entries[0].el);
    if (sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 2) p = entries.length - 1;   // cannot scroll further
    entries.forEach(function (e, k) {
      var t = Math.max(0, 1 - Math.abs(k - p));   // 1 on the reading line, 0 one section away
      var c = accentRGB.map(function (v, j) { return Math.round(v * t + textRGB[j] * (1 - t)); });
      e.a.style.color = "rgb(" + c.join(",") + ")";
      e.a.style.opacity = (0.5 + 0.5 * t).toFixed(3);
    });
    var k0 = Math.floor(p), k1 = Math.min(entries.length - 1, k0 + 1), f = p - k0;
    var r0 = entries[k0].li, r1 = entries[k1].li;
    marker.style.top = (r0.offsetTop * (1 - f) + r1.offsetTop * f) + "px";
    marker.style.height = (r0.offsetHeight * (1 - f) + r1.offsetHeight * f) + "px";
    var cur = entries[Math.round(p)].li;   // keep the current entry in view inside a long list
    if (cur.offsetTop < body.scrollTop + 16 || cur.offsetTop + cur.offsetHeight > body.scrollTop + body.clientHeight - 16) {
      body.scrollTop = cur.offsetTop - body.clientHeight / 2;
    }
  }

  var frame = 0, timer = 0;
  function schedule() { if (!frame) frame = requestAnimationFrame(function () { frame = 0; update(); }); }
  function scheduleRebuild() { clearTimeout(timer); timer = setTimeout(rebuild, 150); }
  doc.addEventListener("scroll", schedule, true);
  window.addEventListener("resize", schedule);
  new MutationObserver(function (records) {
    for (var i = 0; i < records.length; i++) {
      var t = records[i].target;
      if (t === doc.body && records[i].type === "attributes") continue;   // our own class on body
      if (t.nodeType === 1 && t.closest && t.closest("#rl-contents")) continue;
      scheduleRebuild();
      return;
    }
  }).observe(doc.body, { childList: true, subtree: true, attributes: true,
                         attributeFilter: ["hidden", "aria-hidden", "open", "aria-selected"] });
  window.__rlContents = { rebuild: rebuild };
  rebuild();
  setTimeout(rebuild, 500);
  setTimeout(rebuild, 1500);
})();
