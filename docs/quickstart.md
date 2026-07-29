# Quickstart

```python
from pydantic import BaseModel
from runweaver import LocalExecutor, Pipeline, function_block

class Values(BaseModel):
    values: list[float]

def square(inputs: Values, context) -> Values:
    return Values(values=[value**2 for value in inputs.values])

pipeline = Pipeline("quickstart").then(
    function_block(square, inputs=Values, outputs=Values)
)
result = LocalExecutor().run(pipeline, Values(values=[1, 2, 3]))
assert result.final_output.values == [1, 4, 9]
```

For durable mode provide `LocalExecutionConfig(materialization="durable", ...)`.
The block signature stays unchanged. Inspect artifact lineage with
`runweaver lineage ARTIFACT_ID` and resume with the same experiment/plan plus
`resume=True`.

The complete generation → preprocessing → fit → prediction → evaluation
example is Tutorial 1.
