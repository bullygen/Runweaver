# Security policy and trust model

Report vulnerabilities privately to the repository maintainers. Do not include
live credentials or sensitive artifacts in an issue.

## Trust boundaries

- JSON, text, bytes and NumPy serializers do not execute payload code.
- Pickle/cloudpickle artifact loading is not included in core. If an external
  plugin enables it, treat every artifact origin as fully trusted.
- Remote artifact bytes are verified against SHA-256 and a commit manifest.
- Credentials are resolved by secret name at runtime. Config snapshots, logs,
  manifests and MLflow params must contain only `secret://name` references.
- URIs shown to users must mask embedded credentials.
- Plugin discovery uses installed Python entry points; config cannot execute an
  arbitrary source file.
- Subprocess blocks are not enabled by default. An adapter must pass an argument
  vector without `shell=True`, declare side effects and restrict its workdir.
- Content paths are generated from hashes; user artifact names never become
  traversal-capable filesystem paths.

The operator is responsible for database/S3 access controls, encryption,
retention, dependency patching and isolation of untrusted user-authored blocks.


