import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from clean_corpus import dedup_lines, line_key, normalize_line


class DedupLinesTest(unittest.TestCase):
    def test_drops_duplicates_and_letterless_lines_in_order(self):
        # 'Δ' is an untypeable letter, so its line normalizes to nothing.
        lines = ['Reply\n', 'hello world', 'reply', '04.11.2019 | קטגוריה: בלוג',
                 '05.12.2020 | קטגוריה: בלוג', '​', '0', ' hello world ', 'Δ']
        self.assertEqual(list(dedup_lines(lines)),
                         ['Reply', 'hello world', '04.11.2019 | קטגוריה: בלוג'])

    def test_min_words(self):
        self.assertEqual(list(dedup_lines(['one two', 'one two three'], min_words=3)),
                         ['one two three'])

    def test_yields_normalized_lines(self):
        self.assertEqual(list(dedup_lines(['it’s', "it's"])), ["it's"])


class NormalizeLineTest(unittest.TestCase):
    CASES = [
        ('it’s “fine” – ok…', 'it\'s "fine" - ok...'),
        ('naïve café', 'naive cafe'),
        ('ﬁne　text', 'fine text'),
        ('מוּמחֶה בֵּית־סֵפֶר צה״ל ג׳', 'מומחה בית-ספר צה"ל ג\''),
        ('a​b x­y', 'ab xy'),
        ('straße is a word', 'is a word'),
        ('a•b © 2024 🙂', 'a b 2024'),
        ('привет hello', 'hello'),
        ('1/2×4 boards, 8′ 6″', '1/2x4 boards, 8\' 6"'),
    ]

    def test_cases(self):
        for src, expected in self.CASES:
            with self.subTest(src=src):
                self.assertEqual(normalize_line(src), expected)


class LineKeyTest(unittest.TestCase):
    def test_letterless_line_has_no_key(self):
        self.assertIsNone(line_key('0 * ​'))

    def test_case_digits_and_punctuation_ignored(self):
        self.assertEqual(line_key('Reply 12'), line_key('reply'))
        self.assertNotEqual(line_key('reply'), line_key('replies'))


if __name__ == '__main__':
    unittest.main()
