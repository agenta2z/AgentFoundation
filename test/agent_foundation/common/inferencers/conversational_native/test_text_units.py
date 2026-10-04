"""Text sizes in UTF-16 code units: the turn context's and the tool bridge's
cuts keep the longest start or end within a budget."""

from __future__ import annotations

import unittest

from agent_foundation.common.inferencers.agentic_inferencers.conversational_native.context.text_units import (
    utf16_head,
    utf16_len,
    utf16_tail,
)

_EMOJI = "\U0001f600"
_SAMPLES = (
    "plain ascii text",
    _EMOJI * 9,
    f"a{_EMOJI}b{_EMOJI}{_EMOJI}cd{_EMOJI}e",
    f"{_EMOJI}é{_EMOJI}中\U00010348x",
    "",
)


def _longest_head(text: str, size: int) -> str:
    fits = [k for k in range(len(text) + 1) if utf16_len(text[:k]) <= size]
    return text[: max(fits)]


def _longest_tail(text: str, size: int) -> str:
    fits = [k for k in range(len(text) + 1) if utf16_len(text[k:]) <= size]
    return text[min(fits) :]


class TextUnitsTest(unittest.TestCase):
    def test_a_character_outside_the_bmp_is_two_units(self) -> None:
        self.assertEqual(utf16_len("abc"), 3)
        self.assertEqual(utf16_len(_EMOJI * 3), 6)
        self.assertEqual(utf16_len(f"é中{_EMOJI}"), 4)
        self.assertEqual(utf16_len(""), 0)

    def test_a_lone_surrogate_is_one_unit(self) -> None:
        self.assertEqual(utf16_len("a\ud83d"), 2)
        self.assertEqual(utf16_head("a\ud83db", 2), "a\ud83d")
        self.assertEqual(utf16_tail("a\ude00b", 2), "\ude00b")

    def test_bmp_text_is_cut_by_characters(self) -> None:
        self.assertEqual(utf16_head("abcdef", 4), "abcd")
        self.assertEqual(utf16_tail("abcdef", 4), "cdef")

    def test_a_text_within_the_budget_is_whole(self) -> None:
        for text in _SAMPLES:
            size = utf16_len(text)
            self.assertEqual(utf16_head(text, size), text)
            self.assertEqual(utf16_tail(text, size), text)
            self.assertEqual(utf16_head(text, size + 10), text)
            self.assertEqual(utf16_tail(text, size + 10), text)

    def test_astral_text_keeps_every_character_that_fits(self) -> None:
        text = _EMOJI * 3000
        self.assertEqual(utf16_head(text, 3880), _EMOJI * 1940)
        self.assertEqual(utf16_tail(text, 3880), _EMOJI * 1940)

    def test_an_odd_budget_leaves_out_the_character_it_would_split(self) -> None:
        self.assertEqual(utf16_head(_EMOJI * 5, 5), _EMOJI * 2)
        self.assertEqual(utf16_tail(_EMOJI * 5, 5), _EMOJI * 2)
        self.assertEqual(utf16_head(f"ab{_EMOJI}c", 3), "ab")
        self.assertEqual(utf16_tail(f"a{_EMOJI}bc", 3), "bc")

    def test_budget_zero_and_one(self) -> None:
        for text in _SAMPLES:
            self.assertEqual(utf16_head(text, 0), "")
            self.assertEqual(utf16_tail(text, 0), "")
            self.assertEqual(utf16_head(text, -5), "")
            self.assertEqual(utf16_tail(text, -5), "")
        self.assertEqual(utf16_head(_EMOJI * 3, 1), "")
        self.assertEqual(utf16_tail(_EMOJI * 3, 1), "")
        self.assertEqual(utf16_head(f"a{_EMOJI}", 1), "a")
        self.assertEqual(utf16_tail(f"{_EMOJI}a", 1), "a")

    def test_every_cut_is_the_longest_that_fits(self) -> None:
        for text in _SAMPLES:
            for size in range(utf16_len(text) + 2):
                with self.subTest(text=text, size=size):
                    head = utf16_head(text, size)
                    tail = utf16_tail(text, size)
                    self.assertEqual(head, _longest_head(text, size))
                    self.assertEqual(tail, _longest_tail(text, size))
                    self.assertTrue(text.startswith(head))
                    self.assertTrue(text.endswith(tail))
