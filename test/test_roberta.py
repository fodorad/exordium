import unittest
from exordium.text.roberta import RobertaWrapper


class SmileTestCase(unittest.TestCase):

    def setUp(self):
        self.model = RobertaWrapper()

    def test_single_string(self):
        feature = self.model('Welcome, this is an example')
        self.assertEqual(feature.shape, (1, 8, 1024))

    def test_multiple_strings(self):
        feature = self.model(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.'])
        self.assertEqual(feature.shape, (2, 14, 1024))


if __name__ == '__main__':
    unittest.main()