"""
Test Cases for Part 1 (Extraction)

Test coverage:
    - Normal asserts to make sure the function is working
    - Edge cases of missing data, empty content, or non-HTML standard strings
    - Special cases of non readable characters and whitespaces 

# Author: Darwin Zhang
# Date: 2025-12-09
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from src.part_1 import extract_l1, extract_text, parse_html
import unittest

class TestExtraction(unittest.TestCase):

    """
    Test 'extract_l1'
    """
    def test_extract_l1_normal(self):
        # Normal case
        html = "<html><li class='speaking'>Spanish</li></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_l1(soup)

        self.assertEqual(result, "Spanish")

    """
    Test 'extract_text'
    """

    """
    Test 'parse_html'
    """

    """
    Test 'iterate_documents'
    """


if __name__ == "__main__":
    unittest.main()