# Data and model contracts

Pydantic JSON Schema is the machine-readable I/O contract. A port also has a
name, cardinality, semantic role, media types and version compatibility rule.

Use references for arrays, datasets and models that should not cross process
boundaries or enter the database. Core serializers are strict JSON, bytes,
text and NumPy `.npy` with `allow_pickle=False`. Custom serializers declare an
ID, version and media type and pass the serializer contract tests.

Changing a schema or serializer requires a version change. A consumer can
require exact, same-major or documented backward compatibility. Pipeline
validation rejects missing or incompatible required fields before running.
