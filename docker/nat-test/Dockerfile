# Real-NAT test image: a swarmint node OR a NAT gateway, selected at runtime.
FROM python:3.12-slim

# iproute2 (ip route), iptables (MASQUERADE) — needed by the gateway role and
# by node containers that must point their default route at a gateway.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 iptables iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy just what an install needs first (better layer caching).
COPY pyproject.toml README.md requirements.txt /app/
COPY swarmint /app/swarmint
RUN pip install --no-cache-dir -e ".[net]"

COPY docker/nat-test/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
