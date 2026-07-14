"""WebRTC data-channel bridge for an inference query — Phase 2 of browser P2P.

Deliberately NOT a new answering path: a query arriving over an RTCDataChannel
is verified and answered with the exact same functions UDP queries use —
verify_envelope / unpack_inference_query / SwarmNode.answer_query (PURE,
untouched) / pack_inference_response / sign_envelope. The only new code is the
transport (aiortc) and the tiny glue that hands bytes from one to the other.

This node keeps its EXISTING identity (the same node_id it uses on the real UDP
swarm) — a browser querying it over WebRTC is querying the SAME swarm member,
just over a transport a browser can actually open (browsers cannot do raw UDP).
"""

import logging
import time
from collections import OrderedDict

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from . import wire
from .identity import Identity, ReplayGuard, verify_envelope

log = logging.getLogger(__name__)

# A cloud VM's local candidate is its PRIVATE address (GCP 1:1-NATs a public IP
# onto an internal one; the OS never sees the public IP directly). Without a
# STUN server configured HERE — on the answering/Python side, not just the
# browser side — aiortc never discovers its server-reflexive (public) address
# and only offers a useless private-IP candidate, so a remote browser's ICE
# connectivity checks silently time out no matter how well the browser side is
# configured. Mirrors the STUN server infer.html already sets on the browser's
# RTCPeerConnection — both ends must resolve their real public address.
_DEFAULT_ICE_SERVERS = [RTCIceServer(urls="stun:stun.l.google.com:19302")]

RECENT_BROWSER_CLIENTS_MAX = 200  # FIFO cap, mirrors ReplayGuard's bound-under-churn pattern

# node_id -> {"first_seen", "last_seen", "n_queries", "last_label", "last_confidence"}.
# Module-level and deliberately EPHEMERAL: a browser client's identity is a fresh
# random keypair per query by design (infer.html), so this is an activity window
# ("N browsers asked the swarm something recently"), never a persistent registry —
# nothing more identifying than what a query envelope's signature already reveals.
_recent_browser_clients: "OrderedDict[bytes, dict]" = OrderedDict()


def _record_browser_query(sender_id: bytes, label, confidence: float) -> None:
    now = time.time()
    entry = _recent_browser_clients.get(sender_id)
    if entry is None:
        entry = {"first_seen": now, "n_queries": 0}
        if len(_recent_browser_clients) >= RECENT_BROWSER_CLIENTS_MAX:
            _recent_browser_clients.popitem(last=False)  # evict oldest
    else:
        _recent_browser_clients.pop(sender_id, None)  # re-insert to move to the end (LRU-ish)
    entry.update(last_seen=now, n_queries=entry["n_queries"] + 1,
                last_label=label, last_confidence=confidence)
    _recent_browser_clients[sender_id] = entry


def recent_browser_clients(now: float, window_s: float = 300.0) -> list:
    """Browser-answered queries within the last `window_s` — for the status
    page's transparency view ("browser clients appear on the beacon too")."""
    out = []
    for sender_id, entry in _recent_browser_clients.items():
        age_s = now - entry["last_seen"]
        if age_s > window_s:
            continue
        out.append({"id": sender_id.hex(), "age_s": age_s, "n_queries": entry["n_queries"],
                   "last_label": entry["last_label"], "last_confidence": entry["last_confidence"]})
    out.sort(key=lambda c: c["age_s"])
    return out


class WebRTCAnswerer:
    """One RTCPeerConnection, wired to an existing SwarmNode. Construct one per
    incoming browser connection (cheap: aiortc peer connections are lightweight
    until media/data actually flows)."""

    def __init__(self, identity: Identity, node, *, on_query=None):
        self.identity = identity
        self.node = node                      # a SwarmNode — answer_query() is PURE
        self.replay_guard = ReplayGuard()
        self.on_query = on_query               # optional callback(label, confidence) for logging/UI
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=_DEFAULT_ICE_SERVERS))
        self._channel = None

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self._channel = channel

            @channel.on("message")
            def on_message(message):
                self._handle_message(channel, message)

    def _handle_message(self, channel, message: bytes) -> None:
        result = verify_envelope(message, self.replay_guard)
        if not result.ok:
            log.warning("webrtc query rejected: %s", result.reason)
            return
        try:
            kind = wire.peek_kind(result.body)
        except Exception:
            return
        if kind == "query":
            payload = wire.unpack_inference_query(result.body)
            label, confidence = self.node.answer_query(payload["x"])  # PURE, untouched
            _record_browser_query(result.sender, label, confidence)
            if self.on_query is not None:
                self.on_query(result.sender, label, confidence)
            body = wire.pack_inference_response(payload["id"], label, confidence)
        elif kind == "correction":
            payload = wire.unpack_correction_claim(result.body)
            # result.sender is the browser's per-query signing pubkey-derived id --
            # ALWAYS routed through the untrusted-claim gate, never node.observe()
            # or model.learn() directly (see submit_correction_claim's docstring).
            outcome = self.node.submit_correction_claim(payload["x"], payload["label"], result.sender)
            body = wire.pack_correction_ack(payload["id"], outcome["promoted"], outcome["corroborated"])
        else:
            return  # this bridge only answers queries/corrections, nothing else
        envelope = self.identity.sign_envelope(body, ts=time.time())
        channel.send(envelope)

    async def accept_offer(self, sdp: str, sdp_type: str) -> RTCSessionDescription:
        """Standard WebRTC answerer flow: given the browser's offer, return our
        answer. The caller (Phase 3's signaling endpoint) carries the SDP over
        whatever channel the beacon exposes — this class doesn't know or care."""
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    async def close(self) -> None:
        await self.pc.close()
