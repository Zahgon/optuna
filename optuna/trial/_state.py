import enum


class TrialState(enum.IntEnum):

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3
    WAITING = 4

    __str__ = enum.Enum.__str__  # To show the name of the state.

    def is_finished(self) -> bool:
        """Return a bool value to represent whether the trial state is unfinished or not.

        The unfinished state is either ``RUNNING`` or ``WAITING``.
        """

        return self != TrialState.RUNNING and self != TrialState.WAITING
