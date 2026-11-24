from __future__ import annotations

import unittest

from ov_voice.answers import Interrupt, Speak, SpokenAnswers


class SpokenAnswersTest(unittest.TestCase):
    def test_answers_are_spoken_in_turn(self) -> None:
        answers = SpokenAnswers()
        self.assertEqual(answers.answer("Someone fell."), [Speak(1, "Someone fell.")])
        self.assertEqual(answers.spoken(1), [])
        self.assertEqual(answers.answer("He got up."), [Speak(2, "He got up.")])

    def test_a_newer_answer_cuts_the_one_playing(self) -> None:
        answers = SpokenAnswers()
        answers.answer("One.")
        self.assertEqual(
            answers.answer("Two."), [Interrupt(1, by_user=False), Speak(2, "Two.")]
        )
        # The cut answer's playback report comes late and changes nothing.
        self.assertEqual(answers.spoken(1), [])

    def test_the_user_stops_the_answer_and_drops_the_one_waiting(self) -> None:
        answers = SpokenAnswers()
        answers.answer("first")
        self.assertEqual(answers.user_started(), [Interrupt(1, by_user=True)])
        self.assertEqual(answers.user_stopped(), [])

    def test_nothing_is_spoken_over_the_user(self) -> None:
        answers = SpokenAnswers()
        answers.user_started()
        self.assertEqual(answers.answer("old"), [])
        self.assertEqual(answers.answer("late"), [])
        self.assertEqual(answers.user_stopped(), [Speak(1, "late")])


if __name__ == "__main__":
    unittest.main()
