"""Async network-backed distributed inference (T6).

Decision #023 D8 / plan review gap #1: SwarmNode.infer()/calibrate_trust()
call peer.answer_query() directly on in-process objects — that cannot survive
a move to real sockets (calibrate_trust alone does up to 60 sequential calls
per peer). Rather than rip out that working, already-tested in-process path
(still used by the deterministic simulator's `directory` dict), this module
adds the network-capable equivalent: the SAME trust-ranked selection and the
SAME pure functions (SwarmNode.answer_query, aggregate.aggregate, and the
`0.5*t + 0.5*(hits/seen)` trust-update formula) driven by request-collect-
aggregate over a real Bus instead of direct method calls.

answer_query() and aggregate() are imported and called verbatim — zero
changes to either — which is the actual invariant this task had to satisfy.
"""

import asyncio
import os
from dataclasses import dataclass, field

import numpy as np

from ..inference.aggregate import Response, aggregate
from ..network.bus import Message
from ..network.wire import (
    pack_calibration_query, pack_calibration_response, pack_inference_query,
    pack_inference_response, unpack_calibration_query, unpack_calibration_response,
    unpack_inference_query, unpack_inference_response,
)
from ..node.swarm_node import TRUST_INIT

QUERY_TIMEOUT_S = 0.1     # 100ms, per the original spec's inference timeout (§8)
CALIB_CHUNK = 10          # holdout rows per CalibrationQuery datagram


def _new_qid() -> bytes:
    return os.urandom(8)


def _encode_query(payload):
    return pack_inference_query(payload["id"], payload["x"], payload["confidence_threshold"])


def _decode_query(body):
    return unpack_inference_query(body)


def _encode_response(payload):
    return pack_inference_response(payload["id"], payload["label"], payload["confidence"])


def _decode_response(body):
    return unpack_inference_response(body)


def _encode_calib_q(payload):
    return pack_calibration_query(payload["id"], payload["chunk_index"], payload["total_chunks"], payload["xs"])


def _decode_calib_q(body):
    return unpack_calibration_query(body)


def _encode_calib_r(payload):
    return pack_calibration_response(payload["id"], payload["chunk_index"], payload["labels"])


def _decode_calib_r(body):
    return unpack_calibration_response(body)


@dataclass
class InferenceService:
    """Owns the query/response/calibration RPC for ONE node, layered on its
    Bus. Construct via wire_up(), which also multiplexes gossip("protos")
    straight to the node's untouched receive()."""

    node: object
    bus: object
    _pending: dict = field(default_factory=dict)         # qid -> asyncio.Future[(label, conf)]
    _pending_calib: dict = field(default_factory=dict)   # qid -> {expected, labels, got, fut}

    def dispatch(self, msg: Message) -> None:
        if msg.kind == "query":
            self._handle_query(msg)
        elif msg.kind == "response":
            self._handle_response(msg)
        elif msg.kind == "calib_q":
            self._handle_calibration_query(msg)
        elif msg.kind == "calib_r":
            self._handle_calibration_response(msg)

    # ---- answering (server side) ----

    def _handle_query(self, msg: Message) -> None:
        label, conf = self.node.answer_query(msg.payload["x"])  # PURE, untouched
        self.bus.send(Message(self.node.node_id, msg.sender, "response",
                              {"id": msg.payload["id"], "label": label, "confidence": conf}))

    def _handle_calibration_query(self, msg: Message) -> None:
        payload = msg.payload
        labels = [self.node.answer_query(x)[0] for x in payload["xs"]]  # PURE, untouched
        self.bus.send(Message(self.node.node_id, msg.sender, "calib_r",
                              {"id": payload["id"], "chunk_index": payload["chunk_index"], "labels": labels}))

    # ---- querying (client side) ----

    async def query_peer(self, peer_id, x: np.ndarray, confidence_threshold: float = 0.0,
                         timeout: float = QUERY_TIMEOUT_S):
        qid = _new_qid()
        fut = asyncio.get_running_loop().create_future()
        self._pending[qid] = fut
        self.bus.send(Message(self.node.node_id, peer_id, "query",
                              {"id": qid, "x": x, "confidence_threshold": confidence_threshold}))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return (None, 0.0)  # abstain-on-timeout, same shape as an honest abstention
        finally:
            self._pending.pop(qid, None)

    def _handle_response(self, msg: Message) -> None:
        fut = self._pending.get(msg.payload["id"])
        if fut is not None and not fut.done():
            fut.set_result((msg.payload["label"], msg.payload["confidence"]))

    async def infer(self, x: np.ndarray, peer_ids: list, top_n: int = 20,
                    fallback_threshold: float = 0.34, timeout: float = QUERY_TIMEOUT_S):
        """Network-backed equivalent of SwarmNode.infer(): identical trust
        ranking, identical fan-out width, identical PURE aggregate() call —
        the only difference is querying peers over RPC instead of calling
        their answer_query() directly."""
        ranked = sorted(peer_ids, key=lambda p: self.node.trust.get(p, TRUST_INIT), reverse=True)[:top_n]
        if not ranked:
            local = self.node.model.predict(x)
            return (local, 0.0) if local is not None else (None, 0.0)
        results = await asyncio.gather(*(self.query_peer(pid, x, timeout=timeout) for pid in ranked))
        responses = [
            Response(responder=pid, label=label, confidence=conf,
                     trust=self.node.trust.get(pid, TRUST_INIT))
            for pid, (label, conf) in zip(ranked, results)
        ]
        label, score = aggregate(responses)  # PURE, untouched
        if label is None or score < fallback_threshold:
            local = self.node.model.predict(x)
            if local is not None:
                return local, score
        return label, score

    async def calibrate_trust(self, peer_ids: list, timeout: float = QUERY_TIMEOUT_S * 3) -> None:
        """Network-backed equivalent of SwarmNode.calibrate_trust(): same
        reward formula (0.5*t + 0.5*hit-rate), but the querier's holdout is
        sent CHUNKED (decision #023 D8) instead of one query per sample —
        60 samples x 20 peers sequential was the original blocker."""
        if not self.node.holdout_x:
            return
        xs = np.stack(list(self.node.holdout_x))
        ys = list(self.node.holdout_y)
        chunks = [xs[i:i + CALIB_CHUNK] for i in range(0, len(xs), CALIB_CHUNK)]

        async def calibrate_one(pid):
            if pid == self.node.node_id:
                return
            qid = _new_qid()
            fut = asyncio.get_running_loop().create_future()
            self._pending_calib[qid] = {
                "expected": len(chunks), "labels": [None] * len(chunks), "got": 0, "fut": fut,
            }
            for i, chunk in enumerate(chunks):
                self.bus.send(Message(self.node.node_id, pid, "calib_q",
                                      {"id": qid, "chunk_index": i, "total_chunks": len(chunks), "xs": chunk}))
            try:
                all_labels = await asyncio.wait_for(fut, timeout)
            except asyncio.TimeoutError:
                return
            finally:
                self._pending_calib.pop(qid, None)

            flat = [l for chunk_labels in all_labels for l in chunk_labels]
            hits = seen = 0
            for label, y in zip(flat, ys):
                if label is None:
                    continue
                seen += 1
                hits += (label == y)
            if seen:
                t = self.node.trust.get(pid, TRUST_INIT)
                self.node.trust[pid] = 0.5 * t + 0.5 * (hits / seen)  # SAME formula as SwarmNode.calibrate_trust

        await asyncio.gather(*(calibrate_one(pid) for pid in peer_ids))

    def _handle_calibration_response(self, msg: Message) -> None:
        payload = msg.payload
        entry = self._pending_calib.get(payload["id"])
        if entry is None:
            return
        entry["labels"][payload["chunk_index"]] = payload["labels"]
        entry["got"] += 1
        if entry["got"] >= entry["expected"] and not entry["fut"].done():
            entry["fut"].set_result(entry["labels"])


def wire_up(node, bus) -> InferenceService:
    """Build the InferenceService for `node` over `bus`, register its wire
    codecs (if the bus supports registration — SimBus doesn't need it, real
    UdpBus does), and install a small dispatcher that routes gossip("protos")
    straight to node.receive() (untouched) and everything else to the
    service. This intentionally overwrites the auto-registration SwarmNode
    does in __post_init__ — safe, since Bus.register() is just "set the
    current handler for this id"."""
    svc = InferenceService(node=node, bus=bus)
    if hasattr(bus, "register_kind"):
        bus.register_kind("query", _encode_query, _decode_query)
        bus.register_kind("response", _encode_response, _decode_response)
        bus.register_kind("calib_q", _encode_calib_q, _decode_calib_q)
        bus.register_kind("calib_r", _encode_calib_r, _decode_calib_r)

    def dispatch(msg: Message) -> None:
        if msg.kind == "protos":
            node.receive(msg)
        else:
            svc.dispatch(msg)

    bus.register(node.node_id, dispatch)
    return svc
