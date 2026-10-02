import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from clean_corpus import dedup_lines


class DedupLinesTest(unittest.TestCase):
    def test_drops_duplicates_and_letterless_lines_in_order(self):
        lines = ['Reply\n', 'hello world', 'reply', '04.11.2019 | קטגוריה: בלוג',
                 '05.12.2020 | קטגוריה: בלוג', '​', '0', ' hello world ', 'Δ']
        self.assertEqual(list(dedup_lines(lines)),
                         ['Reply', 'hello world', '04.11.2019 | קטגוריה: בלוג', 'Δ'])

    def test_min_words(self):
        self.assertEqual(list(dedup_lines(['one two', 'one two three'], min_words=3)),
                         ['one two three'])


if __name__ == '__main__':
    unittest.main()
