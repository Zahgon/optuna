class OptunaError(Exception):

    pass


class TrialPruned(OptunaError):

    pass


class CLIUsageError(OptunaError):

    pass


class StorageInternalError(OptunaError):

    pass


class DuplicatedStudyError(OptunaError):

    pass


class UpdateFinishedTrialError(OptunaError, RuntimeError):

    pass


class ExperimentalWarning(Warning):

    pass
