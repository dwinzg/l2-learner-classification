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
    
    def test_extract_l1_missing(self):
        # Missing L1, for edge cases 
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_l1(soup)
        self.assertIsNone(result)

    def test_extract_l1_whitespace(self):
        # If there is still whitespace
        html = "<html><li class='speaking'>  Spanish  </li></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_l1(soup)
        self.assertEqual(result, "Spanish")

    """
    Test 'extract_text'
    """
    def test_extract_text_normal(self):
        # Normal case
        html = "<div id='body_show_ori'>Blah Blah Random text.</div>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_text(soup)
        self.assertEqual(result, "Blah Blah Random text.")

    def test_extract_text_missing(self):
        # No <dev>
        html = "<html><body>Random text div</body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_text(soup)
        self.assertEqual(result, "")

    def test_extract_text_whitespace(self):
        # For many white spaces 
        html = "<div id='body_show_ori'>Many    white    space</div>"
        soup = BeautifulSoup(html, 'html.parser')
        result = extract_text(soup)
        self.assertEqual(result, "Many white space")

    """
    Test 'parse_html'
    """
    def test_parse_html_normal(self):
        # Normal working case of splitting HTML, L1, and text 
        html = """
        <html>
        <li class='speaking'>Spanish</li>
        <div id='body_show_ori'>Given Text.</div>
        </html>
        """
        l1, text, filename = parse_html(html, "test.html")
        self.assertEqual(l1, "Spanish")
        self.assertIn("Text", text)
        self.assertEqual(filename, "test.html")

    def test_parse_html_missing_l1(self):
        # For missing case of L1, but text is present 
        html = """
        <html>
        <div id='body_show_ori'>Given Another Text.</div>
        </html>
        """
        l1, text, filename = parse_html(html, "test.html")
        self.assertIsNone(l1)
        self.assertIn("Another", text)
        self.assertEqual(filename, "test.html")

    def test_parse_html_empty(self):
        # Empty HTML
        html = """<html></html>"""
        l1, text, filename = parse_html(html, "test.html")
        self.assertIsNone(l1)
        self.assertIn(text, '')
        self.assertEqual(filename, "test.html")

if __name__ == "__main__":
    unittest.main()