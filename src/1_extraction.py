"""
## Part 1: Text extraction from HTML

Processing and extracting files for:
    - the native language of the writer (L1)
    - the raw text (string) of the entry, with HTML removed
    - the original filename (so that you can use our provided train/dev/test split) 

# Author: Darwin Zhang
# Date: 2025-12-09
"""

import zipfile 
from bs4 import BeautifulSoup


def extract_l1(soup):
    """
    Extract the native language of the writer (L1)
    <li class='speaking' data-title='Native language' rel='tooltip' title='Native language'>Spanish</li>
    
    Returns:
        str: Native Language
    """
    
    # String extract for the li class
    li_tags = soup.find_all('li')

    for li in li_tags:
        # Check for 'speaking'
        if 'speaking' in li.get('class', []):
            return li.get_text()

    # If not listed 
    return None

def extract_text(soup):
    """
    Extracting the text from HTML structure

    
    Returns:
        str: Text
    """