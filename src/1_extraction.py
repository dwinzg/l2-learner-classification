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
    
    Args:
        soup: bs4 parsed HTML

    Returns:
        str: Native Language
    """
    # String extract for the li class
    li_tags = soup.find_all('li')

        # Check for 'speaking'
    for li in li_tags:
        if 'speaking' in li.get('class', []):
            return li.get_text().strip()

    # If not listed 
    return None

def extract_text(soup):
    """
    Extracting the text from HTML structure
    <div id='body_show_ori'> ... </div>

    Args:
    soup: bs4 parsed HTML

    Returns:
        str: Text
    """
    # Find in <div> and 'body_show_ori'
    text_div = soup.find('div', id='body_show_ori')

    if text_div:    
        # Get all text and extract
        text_get = text_div.get_text(separator=' ', strip=True)
        # Remove white spaces
        text = ' '.join(text_get.split())    
        return text
    
    return ""

def parse_html(html_text, filename):
    """
    Extract original filename 
    (so that you can use our provided train/dev/test split) 

    Args:
        html_text: HTML from files 
        str: Filename 

    Returns:
        tuple: (l1, text, filename)
    """
    soup = BeautifulSoup(html_text, 'html.parser')

    l1 = extract_l1(soup)
    text = extract_text(soup)

    return l1, text, filename 

def iterate_documents(zip_path):
    """
    Go through all fiels and extract the data

    Yields:
        tuple: (l1, text, filename)
    """
    # Open zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        html_files = [f for f in zf.namelist() if f.endswith('.html')]
        
        # Loop through each file
        for filename in html_files:
            # Read file
            with zf.open(filename) as f:
                html_content = f.read().decode('utf-8', errors='ignore')
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            # Extract L1 and text
            l1 = extract_l1(soup)
            text = extract_text(soup)
            # Yield document
            yield l1, text, filename
