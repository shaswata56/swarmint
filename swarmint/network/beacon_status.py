"""A tiny, dependency-free HTTP status page for the public beacon (rendezvous).

WHY: the beacon is the one publicly-reachable point in an otherwise peer-to-peer
swarm, so it is the natural place to make the network OBSERVABLE and honest —
anyone can browse beacon.swarmint.org and see who is connected, the public
(post-NAT) endpoint the beacon actually observes for each peer, whether a direct
hole-punched path was established, and how fresh each peer is. Transparency, not
control: the page is read-only and reports only what the rendezvous already sees
in the course of doing its job (it invents no data and claims no NAT type it
cannot substantiate from observation).

HOW: a minimal asyncio HTTP/1.1 responder (no framework, no new dependency, in
keeping with the project's small-surface ethos) that renders a live snapshot of
the rendezvous NetNode on each GET. Self-contained HTML; meta-refreshes every 5s.
"""

import asyncio
import html
import time


def _fmt_ago(seconds: float) -> str:
    if seconds < 1:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h ago"


# active if the beacon has heard from the peer within this window
ACTIVE_WINDOW_S = 30.0


def snapshot(net, start_time: float, now: float) -> dict:
    """Read-only WHOLE-SWARM view. Each peer row aggregates:
      - THIS beacon's directly-known peers (discovery/bus.peer_addrs + freshness), and
      - peers reported by OTHER beacons via the signed P2P census gossip (no HTTP).
    Merged by node_id, so every beacon renders the same complete swarm-wide peer
    list. Columns are only ones meaningful ACROSS beacons — endpoint, topics, which
    beacons vouch for the peer ("seen via"), freshness — deliberately NOT the
    observer-relative NAT/hole-punch facts (those mean nothing in another beacon's
    aggregate)."""
    disc = net.discovery
    bus = net.bus
    fed = getattr(net, "federation", None)
    own_name = (fed.own.get("name") or "this beacon") if fed is not None else "this beacon"

    # --- this beacon's own peers (excluding other beacons, which have their own row
    #     in the Federated Beacons table) ---
    beacon_ids = set(fed.registry.beacons) if fed is not None else set()
    local = {}
    for nid in set(disc.peer_addrs) | set(bus.peer_addrs):
        if nid in beacon_ids or nid == net.identity.node_id:
            continue
        addr = disc.peer_addrs.get(nid) or bus.peer_addrs.get(nid)
        seen = disc.peer_seen_at.get(nid)
        age = (now - seen) if seen else None
        local[nid.hex()] = {
            "id": nid.hex(),
            "endpoint": f"{addr[0]}:{addr[1]}" if addr else "-",
            "topics": sorted(disc.peer_topics.get(nid, set())),
            "age_s": age,
            "sources": [own_name],
            "local": True,
        }

    # --- peers reported by other beacons (the remote half of the aggregate) ---
    peers = dict(local)
    if fed is not None:
        for pid_hex, rec in fed.swarm_snapshot(now).items():
            names = [fed.beacon_name(b) for b in rec["sources"]]
            if pid_hex in peers:
                # already known locally — just credit the other beacons that see it too
                for nm in names:
                    if nm not in peers[pid_hex]["sources"]:
                        peers[pid_hex]["sources"].append(nm)
                if not peers[pid_hex]["topics"]:
                    peers[pid_hex]["topics"] = list(rec["topics"])
            else:
                peers[pid_hex] = {
                    "id": pid_hex, "endpoint": rec["ep"], "topics": list(rec["topics"]),
                    "age_s": rec["age_s"], "sources": names, "local": False,
                }

    peer_list = list(peers.values())
    for p in peer_list:
        p["active"] = (p["age_s"] is not None and p["age_s"] <= ACTIVE_WINDOW_S)
    peer_list.sort(key=lambda p: (not p["active"], p["age_s"] if p["age_s"] is not None else 1e9))

    federation = fed.registry.snapshot(now) if fed is not None else []
    return {
        "beacon_id": net.identity.node_id.hex(),
        "uptime_s": now - start_time,
        "n_total": len(peer_list),
        "n_active": sum(1 for p in peer_list if p["active"]),
        "peers": peer_list,
        "federates": fed is not None,
        "federation": federation,
        "n_beacons_reachable": sum(1 for b in federation if b["reachable"]),
    }


def render_html(snap: dict) -> str:
    """Render the LIVE status shell. The page is a static shell that renders from
    an embedded initial snapshot (instant first paint), then polls /status.json and
    patches the DOM in place — keyed row updates, locally-ticking ages/uptime, and a
    localStorage cache — so the screen updates smoothly via the API with no reload
    or flicker (replacing the old full-page meta-refresh)."""
    import json
    # Embed the freshest snapshot for an instant first paint; </ escaped so a value
    # can never break out of the <script>. All values are re-inserted via textContent
    # on the client, so there is no HTML-injection surface.
    initial = json.dumps(snap).replace("<", "\\u003c")
    return _PAGE.replace("/*__INITIAL__*/null", initial)


_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>swarmint beacon — live peers</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230b1020'/%3E%3Cg stroke='%236a93ff' stroke-width='3' stroke-linecap='round' opacity='0.8'%3E%3Cline x1='32' y1='18' x2='17' y2='45'/%3E%3Cline x1='32' y1='18' x2='47' y2='45'/%3E%3Cline x1='17' y1='45' x2='47' y2='45'/%3E%3C/g%3E%3Ccircle cx='32' cy='18' r='7' fill='%23cdd8ff'/%3E%3Ccircle cx='17' cy='45' r='6' fill='%236a93ff'/%3E%3Ccircle cx='47' cy='45' r='6' fill='%236a93ff'/%3E%3C/svg%3E">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0b0e14; color:#d7dce5;
         font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:22px; margin:0 0 4px; letter-spacing:-.02em; }
  h1 .g { color:#5ee0a0; }
  .live { font-size:12px; color:#8b93a7; margin-left:8px; font-weight:400; vertical-align:2px; }
  .livedot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px;
             background:#5ee0a0; box-shadow:0 0 8px #5ee0a099; transition:background .3s ease; }
  .livedot.stale { background:#e0b25e; box-shadow:0 0 8px #e0b25e99; }
  .sub { color:#8b93a7; margin:0 0 24px; font-size:14px; }
  .sub a { color:#7fb2ff; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }
  .card { background:#141926; border:1px solid #222a3a; border-radius:12px;
           padding:14px 18px; min-width:120px; }
  .card .n { font-size:26px; font-weight:600; transition:color .3s ease; }
  .card .l { color:#8b93a7; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
  .card.hi .n { color:#5ee0a0; }
  h2 { font-size:16px; margin:34px 0 6px; letter-spacing:-.01em; }
  .tblwrap { overflow-x:auto; border:1px solid #222a3a; border-radius:12px; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  th,td { text-align:left; padding:10px 14px; border-bottom:1px solid #1b2231; white-space:nowrap; }
  th { color:#8b93a7; font-weight:500; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  tr:last-child td { border-bottom:none; }
  tbody tr { transition: opacity .35s ease, background-color .6s ease; }
  tr.inactive { opacity:.5; }
  tr.enter { opacity:0; }
  tr.flash { background-color:#16233a; }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
  a.idc { color:#7fb2ff; text-decoration:none; }
  a.idc:hover { text-decoration:underline; }
  a.idc.nolink { color:inherit; cursor:default; pointer-events:none; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px;
         transition:background .3s ease, box-shadow .3s ease; }
  .dot.active { background:#5ee0a0; box-shadow:0 0 8px #5ee0a099; }
  .dot.inactive { background:#5a6376; }
  .pill { padding:2px 9px; border-radius:20px; font-size:12px; transition:background .3s ease, color .3s ease; }
  .nat-public { background:#12351f; color:#7ff0a8; }
  .nat-nat { background:#2a2136; color:#c9a6ff; }
  .nat-unknown { background:#22272f; color:#8b93a7; }
  .empty { text-align:center; color:#8b93a7; padding:28px 14px; white-space:normal; }
  footer { color:#5a6376; font-size:12px; margin-top:22px; }
  footer code { color:#8b93a7; }
</style></head>
<body><div class="wrap">
  <h1><span class="g">swarmint</span> beacon<span class="live"><span class="livedot" id="livedot"></span><span id="livetxt">live</span></span></h1>
  <p class="sub">Public rendezvous &amp; relay for the swarmint P2P network — read-only, live.
  Learn more at <a href="https://swarmint.org">swarmint.org</a>.</p>
  <div class="cards">
    <div class="card hi"><div class="n" id="n_active">—</div><div class="l">active peers</div></div>
    <div class="card"><div class="n" id="n_total">—</div><div class="l">known peers</div></div>
    <div class="card"><div class="n" id="uptime">—</div><div class="l">beacon uptime</div></div>
    <div class="card" id="fedcard" style="display:none"><div class="n" id="n_fed">—</div><div class="l">federated beacons</div></div>
  </div>
  <p class="sub" style="margin:0 0 10px;">Every node in the swarm, aggregated from all
  beacons over signed P2P gossip (no central server) — the same whole-swarm view from any beacon.</p>
  <div class="tblwrap"><table>
    <thead><tr>
      <th>peer id</th><th>status</th><th>endpoint</th>
      <th>topics</th><th>seen via</th><th>last seen</th>
    </tr></thead>
    <tbody id="peers"></tbody>
  </table></div>
  <div id="fedsec" style="display:none">
    <h2>Federated beacons</h2>
    <p class="sub" style="margin-bottom:14px;">Other beacons in the mesh — a decentralized, gossiped
    directory (no master). &ldquo;reachable&rdquo; means this beacon confirmed the other's advertised
    address answers a signed probe. Every beacon holds this same view.</p>
    <div class="tblwrap"><table>
      <thead><tr>
        <th>beacon id</th><th>name</th><th>advertised endpoint</th>
        <th>task</th><th>reachability</th><th>last seen</th>
      </tr></thead>
      <tbody id="beacons"></tbody>
    </table></div>
  </div>
  <footer>
    beacon <code class="mono" id="beaconid">—</code> ·
    &ldquo;seen via&rdquo; lists the beacons that vouch for each peer · peers aggregated swarm-wide over
    signed P2P gossip · updates live via the API (no reload) · cached client-side · no data stored.
  </footer>
</div>
<noscript><p style="color:#8b93a7;padding:0 20px">This live view needs JavaScript. The raw data is at
<a href="/status.json" style="color:#7fb2ff">/status.json</a>.</p></noscript>
<script>
window.__INITIAL__ = /*__INITIAL__*/null;
(function(){
  var REFRESH_MS = 5000, CACHE_KEY = "swarmint_status_v1";
  var state = null, baseMs = 0;
  function $(id){ return document.getElementById(id); }
  function setText(node, txt){ if(node && node.textContent !== txt) node.textContent = txt; }
  function setClass(node, cls){ if(node) node.className = cls; }
  function truncId(h){ return (h||"").slice(0,16) + "\\u2026"; }
  function fmtAgo(s){
    if(s == null) return "\\u2014";
    if(s < 1) return "just now";
    if(s < 60) return Math.floor(s) + "s ago";
    if(s < 3600) return Math.floor(s/60) + "m ago";
    return Math.floor(s/3600) + "h ago";
  }
  function fmtUp(s){
    s = Math.max(0, s|0);
    if(s < 60) return s + "s";
    if(s < 3600) return (s/60|0) + "m";
    if(s < 86400) return (s/3600|0) + "h";
    return (s/86400|0) + "d";
  }
  function tr(html){ var t = document.createElement("tr"); t.innerHTML = html; return t; }

  function peerRow(){ return tr(
    '<td class="mono idc"></td>' +
    '<td><span class="dot"></span><span class="statt"></span></td>' +
    '<td class="mono ep"></td>' +
    '<td class="mono top"></td><td class="via"></td><td class="age"></td>'); }
  function updPeer(row, p){
    if(!row) return;
    row.classList.toggle("inactive", !p.active);
    setText(row.querySelector(".idc"), truncId(p.id));
    setClass(row.querySelector(".dot"), "dot " + (p.active ? "active" : "inactive"));
    setText(row.querySelector(".statt"), p.active ? "active" : "inactive");
    setText(row.querySelector(".ep"), p.endpoint);
    setText(row.querySelector(".top"), (p.topics && p.topics.length) ? p.topics.join(",") : "\\u2014");
    var via = (p.sources && p.sources.length) ? p.sources.join(", ") : "\\u2014";
    setText(row.querySelector(".via"), via);
    row.dataset.age = (p.age_s == null ? "" : p.age_s);
    setText(row.querySelector(".age"), fmtAgo(p.age_s));
  }
  function beaconRow(){ return tr(
    '<td><a class="mono idc nolink" target="_blank" rel="noopener"></a></td><td class="nm"></td><td class="mono ep"></td>' +
    '<td class="tk"></td><td><span class="pill re"></span></td><td class="age"></td>'); }
  function updBeacon(row, b){
    if(!row) return;
    var idc = row.querySelector(".idc");
    setText(idc, truncId(b.id));
    if(idc){
      if(b.status_url){ idc.href = b.status_url; idc.classList.remove("nolink"); idc.title = "Open this beacon's status page"; }
      else { idc.removeAttribute("href"); idc.classList.add("nolink"); idc.title = ""; }
    }
    setText(row.querySelector(".nm"), b.name || "\\u2014");
    setText(row.querySelector(".ep"), b.endpoint || (b.host + ":" + b.gossip_port));
    setText(row.querySelector(".tk"), (b.task || "\\u2014") + (b.n_classes ? " \\u00b7 " + b.n_classes + " cls" : "") + (b.same_space ? "" : " \\u00b7 other space"));
    var re = row.querySelector(".re");
    setClass(re, "pill " + (b.reachable ? "nat-public" : "nat-unknown"));
    setText(re, b.reachable ? "\\u2713 reachable" : "unverified");
    row.dataset.age = (b.age_s == null ? "" : b.age_s);
    setText(row.querySelector(".age"), fmtAgo(b.age_s));
  }
  function reconcile(tbody, items, build, upd, cols, emptyHtml){
    var existing = {}, i, r;
    var rows = tbody.querySelectorAll("tr[data-id]");
    for(i=0;i<rows.length;i++) existing[rows[i].dataset.id] = rows[i];
    var ph = tbody.querySelector(".empty-row");
    var wanted = {}, ordered = [];
    for(i=0;i<items.length;i++){
      var k = String(items[i].id); wanted[k] = 1;
      r = existing[k];
      if(!r){ r = build(); r.dataset.id = k; r.classList.add("enter"); }
      upd(r, items[i]); ordered.push(r);
    }
    for(var k2 in existing){ if(!wanted[k2]) existing[k2].remove(); }
    for(i=0;i<ordered.length;i++) tbody.appendChild(ordered[i]);   // append moves nodes in order
    requestAnimationFrame(function(){ for(i=0;i<ordered.length;i++) ordered[i].classList.remove("enter"); });
    if(items.length === 0 && !tbody.querySelector(".empty-row")){
      tbody.appendChild(tr('<td class="empty" colspan="'+cols+'">'+emptyHtml+'</td>')).classList.add("empty-row");
    } else if(items.length > 0 && ph){ ph.remove(); }
  }
  function render(d){
    setText($("n_active"), d.n_active); setText($("n_total"), d.n_total);
    setText($("uptime"), fmtUp(d.uptime_s)); setText($("beaconid"), truncId(d.beacon_id));
    reconcile($("peers"), d.peers || [], peerRow, updPeer, 6,
      "No peers in the swarm yet. Start a node pointed at any beacon and it will appear here.");
    var showFed = !!d.federates;
    $("fedsec").style.display = showFed ? "" : "none";
    $("fedcard").style.display = showFed ? "" : "none";
    if(showFed){
      setText($("n_fed"), d.n_beacons_reachable || 0);
      reconcile($("beacons"), d.federation || [], beaconRow, updBeacon, 6,
        "No other beacons known yet. Point a beacon at this one with <code>--genesis</code> and it appears here once reachable.");
    }
  }
  function tickAges(){
    if(!state) return;
    var dt = (Date.now() - baseMs) / 1000, rows = document.querySelectorAll("tr[data-age]"), i, b;
    for(i=0;i<rows.length;i++){ b = rows[i].dataset.age; if(b === "") continue;
      setText(rows[i].querySelector(".age"), fmtAgo(parseFloat(b) + dt)); }
    if(state.uptime_s != null) setText($("uptime"), fmtUp(state.uptime_s + dt));
  }
  function apply(d){ state = d; baseMs = Date.now(); render(d); }
  function setLive(ok){
    var dot = $("livedot"); if(dot) dot.classList.toggle("stale", !ok);
    setText($("livetxt"), ok ? "live" : "reconnecting\\u2026");
  }
  function poll(){
    fetch("/status.json", {cache:"no-store"}).then(function(r){
      if(!r.ok) throw 0; return r.json();
    }).then(function(d){
      apply(d); setLive(true);
      try { localStorage.setItem(CACHE_KEY, JSON.stringify(d)); } catch(e){}
    }).catch(function(e){ setLive(false); if(window.console) console.error("swarmint status poll failed:", e); });
  }
  // Instant first paint: embedded snapshot (freshest) else the localStorage cache.
  var init = window.__INITIAL__;
  if(!init){ try { init = JSON.parse(localStorage.getItem(CACHE_KEY) || "null"); } catch(e){} }
  if(init) apply(init);
  poll();
  setInterval(poll, REFRESH_MS);
  setInterval(tickAges, 1000);
})();
</script>
</body></html>"""


async def serve(net, host: str, port: int, start_time: float) -> asyncio.AbstractServer:
    """Start the status HTTP server bound to (host, port). Returns the server so
    the caller can close it on shutdown. Handles only GET; everything else 404s."""
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            # drain headers (we don't use them) up to a blank line
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in (b"\r\n", b"\n", b""):
                    break
            parts = request_line.decode("latin-1", "replace").split()
            method, path = (parts[0], parts[1]) if len(parts) >= 2 else ("GET", "/")
            clean_path = path.split("?")[0]
            if method == "GET" and clean_path in ("/status.json", "/federation.json"):
                import json
                snap = snapshot(net, start_time, time.time())
                if clean_path == "/federation.json":
                    # Back-compat subset consumed by `swarmint beacons`.
                    body = {"beacon_id": snap["beacon_id"],
                            "n_beacons_reachable": snap.get("n_beacons_reachable", 0),
                            "federation": snap.get("federation", [])}
                else:
                    body = snap  # full live snapshot for the page's client-side updates
                payload = json.dumps(body).encode("utf-8")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Access-Control-Allow-Origin: *\r\n"
                             b"Content-Length: %d\r\nCache-Control: no-store\r\n"
                             b"Connection: close\r\n\r\n" % len(payload) + payload)
            elif method != "GET" or clean_path not in ("/", "/index.html"):
                payload = b"not found"
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n"
                             b"Content-Length: %d\r\nConnection: close\r\n\r\n%s"
                             % (len(payload), payload))
            else:
                page = render_html(snapshot(net, start_time, time.time())).encode("utf-8")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                             b"Content-Length: %d\r\nCache-Control: no-store\r\n"
                             b"Connection: close\r\n\r\n" % len(page) + page)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    return await asyncio.start_server(handle, host, port)
