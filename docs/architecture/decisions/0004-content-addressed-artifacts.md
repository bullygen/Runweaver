# ADR 0004: Content-address artifacts

Status: accepted

## Context

Path existence cannot prove integrity or distinguish identical results.

## Decision

Artifacts use SHA-256 object addresses and a committed manifest. SQL registers
the reference only after the artifact commit.

## Alternatives

Mutable run folders; database blobs.

## Consequences

Deduplication, verification and cache lineage become deterministic. Hashing
cost is paid once at commit and recorded in the manifest.
