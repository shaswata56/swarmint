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

from aiortc import RTCPeerConnection, RTCSessionDescription

from . import wire
from .identity import Identity, ReplayGuard, verify_envelope

log = logging.getLogger(__name__)


class WebRTCAnswerer:
    """One RTCPeerConnection, wired to an existing SwarmNode. Construct one per
    incoming browser connection (cheap: aiortc peer connections are lightweight
    until media/data actually flows)."""

    def __init__(self, identity: Identity, node, *, on_query=None):
        self.identity = identity
        self.node = node                      # a SwarmNode — answer_query() is PURE
        self.replay_guard = ReplayGuard()
        self.on_query = on_query               # optional callback(label, confidence) for logging/UI
        self.pc = RTCPeerConnection()
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
        if kind != "query":
            return  # this bridge only answers inference queries, nothing else
        payload = wire.unpack_inference_query(result.body)
        label, confidence = self.node.answer_query(payload["x"])  # PURE, untouched
        if self.on_query is not None:
            self.on_query(label, confidence)
        body = wire.pack_inference_response(payload["id"], label, confidence)
        envelope = self.identity.sign_envelope(body, ts=__import__("time").time())
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
