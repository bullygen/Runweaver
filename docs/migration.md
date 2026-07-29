# Migration guide

Start with observable behavior: entry points, stage order, I/O, seeds,
intermediate artifacts, partial failures and selection rules. Add small golden
or statistical characterization tests before moving code.

Extract computation from global config and disk writes. Give it explicit
Pydantic input/output and wrap it as a block. Keep adapters until end-to-end
comparison passes. Replace manual loops with planners, pools with executors,
folders/status JSON with stores, ranking with decisions, and local candidate
logic with refinement.

The repository-specific audit, plan and compatibility table are under
`migration/`. Tutorial 12 is the executable article v1/v2 migration example.
