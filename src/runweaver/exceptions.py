"""Public exception hierarchy for Runweaver."""


class PipelineError(Exception):
    """Base class for all library errors."""


class PipelineValidationError(PipelineError):
    """A pipeline graph or block declaration is invalid."""


class PortCompatibilityError(PipelineValidationError):
    """Connected input and output schemas are incompatible."""


class ConfigurationError(PipelineError):
    """Experiment configuration cannot be validated or resolved."""


class PlanningError(PipelineError):
    """A planner cannot produce a valid immutable design."""


class ExecutionError(PipelineError):
    """Execution failed outside an individual block."""


class BlockExecutionError(ExecutionError):
    """A block failed."""


class RetryableBlockError(BlockExecutionError):
    """A block failure may be retried under its retry policy."""


class NonRetryableBlockError(BlockExecutionError):
    """A block failure must not be retried automatically."""


class CheckpointError(PipelineError):
    """Checkpoint persistence failed."""


class CheckpointCompatibilityError(CheckpointError):
    """A checkpoint is not compatible with the current run."""


class ArtifactError(PipelineError):
    """Artifact persistence failed."""


class ArtifactCorruptionError(ArtifactError):
    """Artifact content does not match its committed manifest."""


class StateTransitionError(PipelineError):
    """A requested domain state transition is not allowed."""


class BackendUnavailableError(PipelineError):
    """An optional execution or tracking backend is unavailable."""


class CancellationRequested(PipelineError):
    """Cooperative cancellation was requested."""


class ExperimentPruned(PipelineError):
    """A planner or pruner stopped a trial early."""
