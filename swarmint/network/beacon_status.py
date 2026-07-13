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
    """Read-only view of the rendezvous's live peer knowledge. Combines:
      - discovery.peer_addrs  : the peer's SELF-ADVERTISED gossip address
      - bus.peer_addrs        : the source address the beacon OBSERVED packets
                                arriving from = the peer's public (post-NAT) mapping
      - discovery.peer_seen_at: last time we heard from the peer (freshness)
      - nat.reachable         : peers a direct (hole-punched) path was opened to
    A peer whose observed public host differs from its advertised host is behind a
    NAT that remapped it — reported plainly as "behind NAT", else "public"."""
    disc = net.discovery
    bus = net.bus
    reachable = getattr(net.nat, "reachable", set()) if net.nat else set()
    peers = []
    for nid in set(disc.peer_addrs) | set(bus.peer_addrs):
        advertised = disc.peer_addrs.get(nid)
        observed = bus.peer_addrs.get(nid)
        seen = disc.peer_seen_at.get(nid)
        age = (now - seen) if seen else None
        adv_host = advertised[0] if advertised else None
        obs_host = observed[0] if observed else None
        # honest classification from what we can observe:
        if observed is None:
            nat = "unknown"
        elif adv_host in (None, "0.0.0.0") or (obs_host and adv_host and obs_host != adv_host):
            nat = "behind NAT"
        else:
            nat = "public"
        peers.append({
            "id": nid.hex(),
            "advertised": f"{advertised[0]}:{advertised[1]}" if advertised else "-",
            "observed": f"{observed[0]}:{observed[1]}" if observed else "-",
            "nat": nat,
            "topics": sorted(disc.peer_topics.get(nid, set())),
            "age_s": age,
            "active": (age is not None and age <= ACTIVE_WINDOW_S),
            "punched": nid in reachable,
        })
    peers.sort(key=lambda p: (not p["active"], p["age_s"] if p["age_s"] is not None else 1e9))
    # Beacon federation (if this beacon participates): the directory of OTHER
    # beacons it knows, each with live reachability. Same soft-state every beacon
    # holds, so this table looks the same viewed from any beacon in the mesh.
    federation = []
    if getattr(net, "federation", None) is not None:
        federation = net.federation.registry.snapshot(now)
    return {
        "beacon_id": net.identity.node_id.hex(),
        "uptime_s": now - start_time,
        "n_total": len(peers),
        "n_active": sum(1 for p in peers if p["active"]),
        "peers": peers,
        "federation": federation,
        "n_beacons_reachable": sum(1 for b in federation if b["reachable"]),
    }


def render_html(snap: dict) -> str:
    def esc(s):
        return html.escape(str(s))

    rows = []
    for p in snap["peers"]:
        dot = "active" if p["active"] else "inactive"
        status = "active" if p["active"] else "inactive"
        nat_cls = {"public": "nat-public", "behind NAT": "nat-nat"}.get(p["nat"], "nat-unknown")
        punched = "✓ direct" if p["punched"] else "via relay/none"
        age = _fmt_ago(p["age_s"]) if p["age_s"] is not None else "—"
        topics = ",".join(map(str, p["topics"])) or "—"
        rows.append(
            f"<tr class='{dot}'>"
            f"<td class='mono'>{esc(p['id'][:16])}…</td>"
            f"<td><span class='dot {dot}'></span>{status}</td>"
            f"<td class='mono'>{esc(p['observed'])}</td>"
            f"<td><span class='pill {nat_cls}'>{esc(p['nat'])}</span></td>"
            f"<td>{esc(punched)}</td>"
            f"<td class='mono'>{esc(topics)}</td>"
            f"<td>{esc(age)}</td>"
            f"</tr>"
        )
    body_rows = "\n".join(rows) or (
        "<tr><td colspan='7' class='empty'>No peers have contacted the beacon yet. "
        "Start a node pointed at this rendezvous and it will appear here.</td></tr>")

    # ---- federated beacons section (only rendered if this beacon federates) ----
    fed_section = ""
    if snap.get("federation") is not None and "n_beacons_reachable" in snap:
        frows = []
        for b in snap["federation"]:
            up_cls = "nat-public" if b["reachable"] else "nat-unknown"
            up_txt = "✓ reachable" if b["reachable"] else "unverified"
            frows.append(
                f"<tr>"
                f"<td class='mono'>{esc(b['id'][:16])}…</td>"
                f"<td>{esc(b['name'] or '—')}</td>"
                f"<td class='mono'>{esc(b['host'])}:{esc(b['gossip_port'])}</td>"
                f"<td>{esc(b['task'] or '—')}</td>"
                f"<td><span class='pill {up_cls}'>{up_txt}</span></td>"
                f"<td>{esc(_fmt_ago(b['age_s']))}</td>"
                f"</tr>")
        frows = "\n".join(frows) or (
            "<tr><td colspan='6' class='empty'>No other beacons known yet. Point a beacon "
            "at this one with <code>--genesis</code> and it will appear here once reachable.</td></tr>")
        fed_section = f"""
  <h2 style="font-size:16px;margin:34px 0 6px;letter-spacing:-.01em;">Federated beacons</h2>
  <p class="sub" style="margin-bottom:14px;">Other beacons in the mesh — a decentralized,
  gossiped directory (no central registry). “reachable” means this beacon confirmed the other's
  advertised address answers a signed probe. Same view from every beacon.</p>
  <div class="tblwrap"><table>
    <thead><tr>
      <th>beacon id</th><th>name</th><th>advertised endpoint</th>
      <th>task</th><th>reachability</th><th>last seen</th>
    </tr></thead>
    <tbody>
{frows}
    </tbody>
  </table></div>"""

    up = _fmt_ago(snap["uptime_s"]).replace(" ago", "")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>swarmint beacon — live peers</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230b1020'/%3E%3Cg stroke='%236a93ff' stroke-width='3' stroke-linecap='round' opacity='0.8'%3E%3Cline x1='32' y1='18' x2='17' y2='45'/%3E%3Cline x1='32' y1='18' x2='47' y2='45'/%3E%3Cline x1='17' y1='45' x2='47' y2='45'/%3E%3C/g%3E%3Ccircle cx='32' cy='18' r='7' fill='%23cdd8ff'/%3E%3Ccircle cx='17' cy='45' r='6' fill='%236a93ff'/%3E%3Ccircle cx='47' cy='45' r='6' fill='%236a93ff'/%3E%3C/svg%3E">
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0b0e14; color:#d7dce5;
         font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.02em; }}
  h1 .g {{ color:#5ee0a0; }}
  .sub {{ color:#8b93a7; margin:0 0 24px; font-size:14px; }}
  .sub a {{ color:#7fb2ff; }}
  .cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:24px; }}
  .card {{ background:#141926; border:1px solid #222a3a; border-radius:12px;
           padding:14px 18px; min-width:120px; }}
  .card .n {{ font-size:26px; font-weight:600; }}
  .card .l {{ color:#8b93a7; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
  .card.hi .n {{ color:#5ee0a0; }}
  .tblwrap {{ overflow-x:auto; border:1px solid #222a3a; border-radius:12px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th,td {{ text-align:left; padding:10px 14px; border-bottom:1px solid #1b2231; white-space:nowrap; }}
  th {{ color:#8b93a7; font-weight:500; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.inactive {{ opacity:.5; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }}
  .dot.active {{ background:#5ee0a0; box-shadow:0 0 8px #5ee0a099; }}
  .dot.inactive {{ background:#5a6376; }}
  .pill {{ padding:2px 9px; border-radius:20px; font-size:12px; }}
  .nat-public {{ background:#12351f; color:#7ff0a8; }}
  .nat-nat {{ background:#2a2136; color:#c9a6ff; }}
  .nat-unknown {{ background:#22272f; color:#8b93a7; }}
  .empty {{ text-align:center; color:#8b93a7; padding:28px 14px; white-space:normal; }}
  footer {{ color:#5a6376; font-size:12px; margin-top:22px; }}
  footer code {{ color:#8b93a7; }}
</style></head>
<body><div class="wrap">
  <h1><span class="g">swarmint</span> beacon</h1>
  <p class="sub">Public rendezvous &amp; relay for the swarmint P2P network — read-only, live.
  Learn more at <a href="https://swarmint.org">swarmint.org</a>.</p>
  <div class="cards">
    <div class="card hi"><div class="n">{snap['n_active']}</div><div class="l">active peers</div></div>
    <div class="card"><div class="n">{snap['n_total']}</div><div class="l">known peers</div></div>
    <div class="card"><div class="n">{esc(up)}</div><div class="l">beacon uptime</div></div>
    {(f'<div class="card"><div class="n">{snap["n_beacons_reachable"]}</div><div class="l">federated beacons</div></div>') if snap.get("federation") is not None else ""}
  </div>
  <div class="tblwrap"><table>
    <thead><tr>
      <th>peer id</th><th>status</th><th>public endpoint (observed)</th>
      <th>reachability</th><th>direct path</th><th>topics</th><th>last seen</th>
    </tr></thead>
    <tbody>
{body_rows}
    </tbody>
  </table></div>
{fed_section}
  <footer>
    beacon <code class="mono">{esc(snap['beacon_id'][:16])}…</code> ·
    “public endpoint” is the post-NAT address the beacon observes packets arriving from ·
    auto-refreshes every 5s · no data is stored beyond the live session.
  </footer>
</div></body></html>"""


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
            if method == "GET" and path.split("?")[0] == "/federation.json":
                import json
                snap = snapshot(net, start_time, time.time())
                payload = json.dumps({"beacon_id": snap["beacon_id"],
                                      "n_beacons_reachable": snap.get("n_beacons_reachable", 0),
                                      "federation": snap.get("federation", [])}).encode("utf-8")
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                             b"Content-Length: %d\r\nCache-Control: no-store\r\n"
                             b"Connection: close\r\n\r\n" % len(payload) + payload)
            elif method != "GET" or path not in ("/", "/index.html"):
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
