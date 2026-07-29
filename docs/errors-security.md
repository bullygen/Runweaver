# Error handling, recovery and security

The public hierarchy distinguishes validation, configuration, planning,
execution, retryable/non-retryable block errors, checkpoint compatibility,
artifact corruption, state transition, backend availability, cancellation and
pruning.

Retries are bounded exponential backoff with jitter and visible events.
Idempotency/side effects gate automatic retry. Recovery validates checkpoint
block, parameters and input-lineage fingerprints.

Core never unpickles artifacts. Secrets use named runtime lookup; remote bytes
are hashed; plugins are entry points; subprocess adapters must avoid shell
string concatenation. See the root security policy for the full trust model.
