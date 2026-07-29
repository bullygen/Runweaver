# Reproducibility

Every random path begins with an explicit experiment/trial seed. `RunContext`
derives child streams without mutable global seeds. Plans record resolved
parameters, replicate index, planner/version, dataset/pipeline version and a
fingerprint.

Lineage records code and environment fingerprints, upstream hashes,
serializer versions and trial identity. Determinism still depends on numerical
libraries, hardware and user blocks; PyTorch utilities can request
deterministic algorithms but may reduce performance or reject unsupported ops.

Never claim bitwise reproduction when the underlying algorithm promises only
statistical equivalence.
