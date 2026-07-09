"""N1: shared genesis embedding distributed over the real networked path — a
joiner fetches the frozen (mean, W) from the rendezvous and reconstructs an
IDENTICAL transform (the shared-space invariant, bootstrapped over UDP)."""

import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarmint.core.embedding import LDAEmbedding
from swarmint.network.identity import Identity
from swarmint.node.net_node import NetNode


async def test_joiner_fetches_identical_embedding_over_udp():
    # fit a genesis embedding (as the rendezvous node would hold it)
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, (6, 40)).astype(np.float32)
    y = np.repeat(range(6), 40)
    X = centers[y] + rng.normal(0, 1.0, (len(y), 40)).astype(np.float32)
    emb = LDAEmbedding().fit(X.astype(np.float32), y)

    id_g, id_j = Identity.generate(), Identity.generate()
    genesis = NetNode(identity=id_g, topic=1, rng=random.Random(1), gossip_interval=10.0, embedding=emb)
    joiner = NetNode(identity=id_j, topic=1, rng=random.Random(2), gossip_interval=10.0)  # no embedding yet

    await genesis.start("127.0.0.1", 0, 0, [])
    dht_g = genesis.discovery.server.transport.get_extra_info("sockname")[1]
    await joiner.start("127.0.0.1", 0, 0, [("127.0.0.1", dht_g)], rendezvous_id=id_g.node_id)

    assert joiner.embedding is None
    fetched = await joiner.fetch_embedding(id_g.node_id, timeout=3.0)

    # the joiner's fetched embedding must produce byte-identical transforms
    probe = X[:20].astype(np.float32)
    same = np.allclose(fetched.transform(probe), emb.transform(probe), atol=1e-5)

    await genesis.stop()
    await joiner.stop()

    assert fetched is not None, "joiner failed to fetch the genesis embedding"
    assert same, "fetched embedding does not reproduce the genesis transform (shared-space broken)"
    assert fetched.out_dim == emb.out_dim


async def _run_all():
    await test_joiner_fetches_identical_embedding_over_udp()
    print("ok  test_joiner_fetches_identical_embedding_over_udp")
    print("all tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all())
