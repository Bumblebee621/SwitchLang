import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from make_chat_testset import orient


class OrientTest(unittest.TestCase):
    def test_reversed_hebrew_is_restored(self):
        vocab = {'הקטן', 'הזה', 'הרגע', 'הציל', 'את', 'חייך'}
        self.assertEqual(orient('ךייח תא ליצה עגרה הזה ןטקה', vocab),
                         'הקטן הזה הרגע הציל את חייך')

    def test_correct_line_unchanged(self):
        self.assertEqual(orient('hello there', {'hello', 'there'}), 'hello there')


if __name__ == '__main__':
    unittest.main()
