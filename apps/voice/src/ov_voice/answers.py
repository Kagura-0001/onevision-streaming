"""Which OV answers to speak.

OV writes each answer back to its memory as soon as it is written, so this
module only decides what the user hears. It knows nothing about the voice
framework around it: the caller reports what happened (an answer finished,
the user started or stopped talking, an answer finished playing) and carries
out the returned actions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speak:
    turn: int
    text: str


@dataclass(frozen=True)
class Interrupt:
    turn: int
    # The user cut in, rather than a newer answer.
    by_user: bool


Action = Speak | Interrupt


class SpokenAnswers:
    """Speaks the newest answer, never over the user.

    OV can answer every second, faster than speech, and an older answer is out
    of date once a newer one is written, so a new answer cuts the one playing.
    The user starting to talk stops the answer and drops the one waiting; an
    answer written while the user talks waits until the turn is over.
    """

    def __init__(self) -> None:
        self._waiting: str | None = None
        self._speaking: int | None = None
        self._user_speaking = False
        self._turns = 0

    def answer(self, text: str) -> list[Action]:
        self._waiting = text
        if self._user_speaking:
            return []
        actions: list[Action] = []
        if self._speaking is not None:
            actions.append(Interrupt(self._speaking, by_user=False))
            self._speaking = None
        return [*actions, *self._next()]

    def spoken(self, turn: int) -> list[Action]:
        if self._speaking != turn:
            return []
        self._speaking = None
        return self._next()

    def user_started(self) -> list[Action]:
        self._user_speaking = True
        self._waiting = None
        turn, self._speaking = self._speaking, None
        return [] if turn is None else [Interrupt(turn, by_user=True)]

    def user_stopped(self) -> list[Action]:
        self._user_speaking = False
        return self._next()

    def _next(self) -> list[Action]:
        if self._speaking is not None or self._user_speaking or self._waiting is None:
            return []
        text, self._waiting = self._waiting, None
        self._turns += 1
        self._speaking = self._turns
        return [Speak(self._turns, text)]
