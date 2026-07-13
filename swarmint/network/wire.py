"""msgpack wire serialization for every message type the swarm sends.

No IDL/codegen (decision #023: protobuf's gencode/runtime version lock is
hostile to a P2P mesh where peers run different builds). Convention instead:
every body carries an explicit WIRE_VERSION int and named short keys; float
vectors always travel as ONE raw bytes blob (np.tobytes), never encoded
element-wise — msgpack floats are 8-9 bytes each, a raw float32 blob is 4
bytes/dim with near-zero framing.

The envelope (pack_envelope/signable_bytes) is crypto-agnostic on purpose:
T3 (identity/signing) computes pubkey/sender/nonce/sig and calls into this
module to assemble the actual bytes-to-sign and the final wire bytes, so the
serialization format has exactly one owner.
"""

from dataclasses import dataclass

import msgpack
import numpy as np

from ..core.prototype_model import Prototype

WIRE_VERSION = 1
MAX_DATAGRAM_BYTES = 1200  # UDP fragmentation-safe budget (spec + decision #023 D9)


def _pack(body: dict) -> bytes:
    return msgpack.packb(body, use_bin_type=True)


def _unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


# ---- envelope (generic transport wrapper; T3 fills pubkey/sender/nonce/sig) ----

def signable_bytes(pubkey: bytes, sender: bytes, ts: float, nonce: bytes, body: bytes) -> bytes:
    """The exact bytes a sender signs and a receiver re-derives to verify —
    everything in the envelope except the signature itself."""
    return _pack({"pk": pubkey, "s": sender, "ts": ts, "n": nonce, "b": body})


def pack_envelope(pubkey: bytes, sender: bytes, ts: float, nonce: bytes,
                  sig: bytes, body: bytes) -> bytes:
    return _pack({"v": WIRE_VERSION, "pk": pubkey, "s": sender, "ts": ts,
                  "n": nonce, "sig": sig, "b": body})


def unpack_envelope(data: bytes) -> dict:
    d = _unpack(data)
    if d.get("v") != WIRE_VERSION:
        raise ValueError(f"unsupported wire version {d.get('v')}")
    return d


def peek_kind(body: bytes) -> str:
    """Read just the `k` discriminator from a message body, so a transport
    can pick the right decoder without hardcoding a kind->decoder table."""
    return _unpack(body)["k"]


# ---- gossip: prototype batches ----

def _pack_proto(p: Prototype) -> dict:
    return {"l": int(p.label), "w": float(p.weight),
            "x": np.asarray(p.vector, dtype=np.float32).tobytes()}


def _unpack_proto(d: dict) -> Prototype:
    vec = np.frombuffer(d["x"], dtype=np.float32).copy()
    return Prototype(vector=vec, label=d["l"], weight=d["w"])


def pack_gossip_protos(protos: list) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "protos", "p": [_pack_proto(p) for p in protos]})


def unpack_gossip_protos(data: bytes) -> list:
    body = _unpack(data)
    assert body["k"] == "protos"
    return [_unpack_proto(d) for d in body["p"]]


# ---- inference RPC ----

def pack_inference_query(qid: bytes, x: np.ndarray, confidence_threshold: float) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "query", "id": qid,
                  "x": np.asarray(x, dtype=np.float32).tobytes(), "t": float(confidence_threshold)})


def unpack_inference_query(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "query"
    return {"id": body["id"], "x": np.frombuffer(body["x"], dtype=np.float32).copy(),
            "confidence_threshold": body["t"]}


def pack_inference_response(qid: bytes, label, confidence: float) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "response", "id": qid,
                  "l": label, "c": float(confidence)})


def unpack_inference_response(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "response"
    return {"id": body["id"], "label": body["l"], "confidence": body["c"]}


# ---- calibration: chunked because a 60-sample holdout doesn't fit one datagram ----

def pack_calibration_query(qid: bytes, chunk_index: int, total_chunks: int, xs: np.ndarray) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "calib_q", "id": qid, "i": chunk_index,
                  "n": total_chunks, "x": np.asarray(xs, dtype=np.float32).tobytes(),
                  "dim": int(xs.shape[1])})


def unpack_calibration_query(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "calib_q"
    flat = np.frombuffer(body["x"], dtype=np.float32)
    xs = flat.reshape(-1, body["dim"]).copy()
    return {"id": body["id"], "chunk_index": body["i"], "total_chunks": body["n"], "xs": xs}


def pack_calibration_response(qid: bytes, chunk_index: int, labels: list) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "calib_r", "id": qid, "i": chunk_index, "l": labels})


def unpack_calibration_response(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "calib_r"
    return {"id": body["id"], "chunk_index": body["i"], "labels": body["l"]}


# ---- discovery: signed rendezvous advert + peer-exchange ----

def pack_advert(node_id: bytes, addr: str, topics, ts: float) -> bytes:
    # topics: iterable of ints (finer topic granularity, D2). A scalar is
    # accepted for backward compatibility.
    topic_list = [int(topics)] if isinstance(topics, int) else sorted(int(t) for t in topics)
    return _pack({"v": WIRE_VERSION, "k": "advert", "id": node_id, "addr": addr,
                  "topics": topic_list, "ts": float(ts)})


def unpack_advert(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "advert"
    return {"node_id": body["id"], "addr": body["addr"],
            "topics": list(body["topics"]), "ts": body["ts"]}


# ---- beacon federation: signed beacon advert + gossip transport + reach probe ----
#
# A BEACON advert is distinct from a peer `advert`: it describes a whole
# rendezvous/relay (a "brain region") — its dial-able public host, BOTH ports, the
# task/space its local swarm runs, and an optional human label — carrying exactly
# what a node on one beacon would need to reach the swarm on another (the seam for
# cross-beacon "one singular system" traffic). It is signed with the beacon's
# identity (sign_envelope), so a forwarding beacon can never fabricate or alter a
# peer beacon's advert: every recipient re-verifies the origin signature itself.

def pack_beacon_advert(node_id: bytes, host: str, gossip_port: int, dht_port: int,
                       task: str, n_classes: int, topics, name: str, ts: float,
                       space_fp: str = "") -> bytes:
    # space_fp: a fingerprint of the shared embedding SPACE this beacon's swarm
    # lives in. Cross-beacon learning is only coherent between beacons with the
    # SAME fingerprint (identical genesis space) — otherwise B's prototype vectors
    # are meaningless to A. Empty string => unknown/legacy (never auto-bridged).
    topic_list = [int(topics)] if isinstance(topics, int) else sorted(int(t) for t in topics)
    return _pack({"v": WIRE_VERSION, "k": "beacon_advert", "id": node_id, "host": host,
                  "gp": int(gossip_port), "dp": int(dht_port), "task": task,
                  "nc": int(n_classes), "topics": topic_list, "name": name,
                  "fp": space_fp, "ts": float(ts)})


def unpack_beacon_advert(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "beacon_advert"
    return {"node_id": body["id"], "host": body["host"], "gossip_port": body["gp"],
            "dht_port": body["dp"], "task": body["task"], "n_classes": body["nc"],
            "topics": list(body["topics"]), "name": body.get("name", ""),
            "space_fp": body.get("fp", ""), "ts": body["ts"]}


def pack_beacon_gossip(payload: dict) -> bytes:
    """Transport for one or more SIGNED beacon-advert envelopes (opaque blobs,
    like `relay`): registration carries the sender's own, gossip carries the set
    it knows. Recipients verify each blob independently. A dict payload so
    UdpBus's prototype list-chunker leaves it alone."""
    return _pack({"v": WIRE_VERSION, "k": "beacon_gossip", "blobs": list(payload["blobs"])})


def unpack_beacon_gossip(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "beacon_gossip"
    return {"blobs": list(body["blobs"])}


def pack_beacon_ping(payload: dict) -> bytes:
    """Reachability probe to a beacon's ADVERTISED address — the live proof its
    public host+port actually accepts inbound (NAT/firewall correct). op = ping|pong."""
    return _pack({"v": WIRE_VERSION, "k": "beacon_ping", "op": payload["op"],
                  "nonce": payload["nonce"]})


def unpack_beacon_ping(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "beacon_ping"
    return {"op": body["op"], "nonce": body["nonce"]}


def pack_pex(peers: list) -> bytes:
    """peers: list of (node_id: bytes, addr: str, topics: iterable[int])."""
    out = []
    for entry in peers:
        nid, addr = entry[0], entry[1]
        topics = entry[2] if len(entry) > 2 else []
        out.append({"id": nid, "addr": addr, "t": sorted(int(t) for t in topics)})
    return _pack({"v": WIRE_VERSION, "k": "pex", "p": out})


def unpack_pex(data: bytes) -> list:
    body = _unpack(data)
    assert body["k"] == "pex"
    return [(d["id"], d["addr"], list(d.get("t", []))) for d in body["p"]]


# ---- genesis embedding distribution (chunked, so a joiner pulls the shared
#      frozen projection from the rendezvous like any other protocol parameter) ----

EMBEDDING_CHUNK_BYTES = 900  # payload bytes per datagram (leaves room for the envelope)


def pack_embedding_request(_payload=None) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "emb_req"})


def unpack_embedding_request(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "emb_req"
    return {}


def pack_embedding_response(payload: dict) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "emb_resp", "i": payload["chunk_index"],
                  "n": payload["total_chunks"], "d": payload["data"]})


def unpack_embedding_response(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "emb_resp"
    return {"chunk_index": body["i"], "total_chunks": body["n"], "data": body["d"]}


# ---- NAT traversal / hole-punching (D3) ----

def pack_nat(payload: dict) -> bytes:
    """Generic NAT-signaling message; payload["op"] discriminates:
    whoami | whoami_resp | connect_req | connect_resp | punch_req | punch | punch_ack.
    A dict payload so UdpBus's prototype list-chunker leaves it alone."""
    return _pack({"v": WIRE_VERSION, "k": "nat", **payload})


def unpack_nat(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "nat"
    return {kk: vv for kk, vv in body.items() if kk not in ("v", "k")}


# ---- relay fallback (T4): reach a peer we can't punch to, via a mutually
#      reachable node (e.g. the rendezvous). `blob` is the ORIGINAL signed
#      envelope, carried untouched so the destination still verifies the true
#      sender; the relay signs the outer wrapper (relay can't forge the inner).

def pack_relay(payload: dict) -> bytes:
    """A -> R: 'please deliver this blob to `target`'. payload: {target: bytes,
    blob: bytes}. A dict so UdpBus's list-chunker leaves it alone."""
    return _pack({"v": WIRE_VERSION, "k": "relay", "t": payload["target"], "b": payload["blob"]})


def unpack_relay(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "relay"
    return {"target": body["t"], "blob": body["b"]}


def pack_relayed(payload: dict) -> bytes:
    """R -> B: 'this blob came from `origin`, relayed by me'. payload:
    {origin: bytes, blob: bytes}."""
    return _pack({"v": WIRE_VERSION, "k": "relayed", "o": payload["origin"], "b": payload["blob"]})


def unpack_relayed(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "relayed"
    return {"origin": body["o"], "blob": body["b"]}


# ---- tamper-evident update chain (D1) ----

def pack_chain_request(_payload=None) -> bytes:
    return _pack({"v": WIRE_VERSION, "k": "chain_req"})


def unpack_chain_request(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "chain_req"
    return {}


CHAIN_ENTRIES_PER_CHUNK = 3  # each ChainEntry is ~210B; 3 + header fits one 1200B datagram


def pack_chain_response(payload: dict) -> bytes:
    # payload: {"chunk_index", "total_chunks", "entries": [ChainEntry.to_wire()...]}.
    # A dict (not a bare list) so UdpBus's prototype list-chunker leaves it
    # alone — chain chunking is done explicitly by the sender with reassembly.
    return _pack({"v": WIRE_VERSION, "k": "chain_resp", "i": payload["chunk_index"],
                  "n": payload["total_chunks"], "e": payload["entries"]})


def unpack_chain_response(data: bytes) -> dict:
    body = _unpack(data)
    assert body["k"] == "chain_resp"
    return {"chunk_index": body["i"], "total_chunks": body["n"], "entries": body["e"]}


@dataclass
class ByteBudget:
    """Measured (not hand-computed) size table — see test_wire.py::test_gossip_batch_fits_budget."""
    dim: int
    n_protos: int
    body_bytes: int
    envelope_overhead_bytes: int
    total_bytes: int

    def fits(self) -> bool:
        return self.total_bytes <= MAX_DATAGRAM_BYTES
