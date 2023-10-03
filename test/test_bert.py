import unittest
from exordium.text.bert import BertWrapper


class SmileTestCase(unittest.TestCase):

    def setUp(self):
        self.model = BertWrapper()

    def test_single_string(self):
        feature = self.model('Welcome, this is an example')
        self.assertEqual(feature.shape, (1, 8, 768))

    def test_multiple_strings(self):
        feature = self.model(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.'])
        self.assertEqual(feature.shape, (2, 14, 768))


if __name__ == '__main__':
    unittest.main()