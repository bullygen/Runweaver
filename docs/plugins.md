# Plugin development

Plugins are normal installed packages with entry points:

```toml
[project.entry-points."runweaver.blocks"]
my-block = "my_package.blocks:factory"
```

Groups also exist for planners, serializers and executors. A block factory
receives a validated parameter dictionary and returns a typed block. Discovery
does not execute arbitrary paths from experiment JSON.

Run the shared contract suites for new artifact/state stores, executors,
trackers, planners, serializers and checkpoint stores. Tutorial 10 implements
and registers a custom block, planner, decision and refinement.
