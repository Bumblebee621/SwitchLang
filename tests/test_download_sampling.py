import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from download_corpora import round_robin, sample_lines, site_of


class SiteOfTest(unittest.TestCase):
    def test_mobile_and_www_labels_dropped(self):
        self.assertEqual(site_of('https://he.m.wikipedia.org/wiki/x'), 'he.wikipedia.org')
        self.assertEqual(site_of('http://www.Example.com/a'), 'example.com')

    def test_missing_url(self):
        self.assertEqual(site_of(None), '')


class RoundRobinTest(unittest.TestCase):
    def test_interleaves_until_all_exhausted(self):
        self.assertEqual(list(round_robin([iter('ab'), iter('xyz')])), list('axbyz'))


class SampleLinesTest(unittest.TestCase):
    def test_site_cap_dedup_and_normalization(self):
        docs = [{'url': 'http://a.com/1', 'text': 'one it’s\ntwo\nthree'},
                {'url': 'http://a.com/2', 'text': 'four\nTWO'},
                {'url': 'http://b.com/1', 'text': 'five'},
                {'url': None, 'text': 'six\nseven\neight'}]  # no URL: exempt from the cap
        self.assertEqual(list(sample_lines(docs, keep_prob=1.0, seed=0, site_cap=2)),
                         ["one it's", 'two', 'five', 'six', 'seven', 'eight'])

    def test_keep_prob_is_seeded(self):
        # Letters only: the dedup key ignores digits, so 'a0'/'a1' would collide.
        docs = [{'url': f'http://s{i}.com', 'text': f'line {chr(97 + i % 26)}{chr(97 + i // 26)}'}
                for i in range(200)]
        first = list(sample_lines(docs, keep_prob=0.3, seed=7, site_cap=10))
        self.assertEqual(first, list(sample_lines(docs, keep_prob=0.3, seed=7, site_cap=10)))
        self.assertTrue(30 < len(first) < 90, len(first))


if __name__ == '__main__':
    unittest.main()
