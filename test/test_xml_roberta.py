import unittest
from exordium.text.xml_roberta import XmlRobertaWrapper


class XmlRobertaTestCase(unittest.TestCase):

    def setUp(self):
        self.model = XmlRobertaWrapper()

    def test_single_string(self):
        feature = self.model('Welcome, this is an example')
        self.assertEqual(feature.shape, (1, 768))

    def test_multiple_strings(self):
        feature = self.model(['Welcome, this is an example', 'An another, longer example. I mean a lot longer.'])
        self.assertEqual(feature.shape, (2, 768))


if __name__ == '__main__':
    unittest.main()