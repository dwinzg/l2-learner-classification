import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from collections import Counter
import string
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



#Test Text
text = "Do you know what the zeitgeist is? This is a term that has been used firstly in Germany by a philosopher J.G. Herder. And it has necessity of being rewritten endlessly like history because it is a kind of word that shows how we interpret that period. So you need to know the situation of the world as it is, to make your own globalized zeitgeist correctly. I also don’t want to use these words that make us (and this article) confused and completely boring. But I just want to introduce this important term and the facts that you would need for making your own values. I have known this term from watching a documentary movie named Zeitgeist that I want to tell about from now on. Here are only three introductions (somewhat changed by myself) of more in the movie. #1. – Religion. Do you believe in God or have religion? I do. But, I don’t believe church and the Bible. That is why I don’t go to church every Sunday, not because of drinking on every Saturday night. This is my own religion. In my opinion, religions are inventions which are designed for making people in a society more clean and kind and for harmonizing them to each others. It is really good goal. But in many cases, the other goals happen. Sometimes the inventions are used for the completely opposite intentions of their infancy . For example, religions can justify horrible wars. Christianity helps this kind of wrong things to be hidden with their HOLY Bible that is a fake according to the docudrama. #2 – Possessions We all have to be free. But, if you feel free continuously without anything, you can even be almost enslaved . So, the thing is, to get actual freedom, we have to know more about what’s going on and have to pursue and find freedom constantly. It seems difficult. Coming to the topic, money makes us free. And also, it makes us enslaved even if we have a lot of it. That is because money is dept. The Bank doesn’t borrow money to us without profit. The docu-drama says that all the money supply on the market is always smaller than the money we have to pay back to the Bank. In other word, to pay back, some always should be bankrupt . Even if that is a story about U.S., nowadays U.S. economy is almost international thing. Who gets bankrupt? Or what nation gets? #3 – Venus project The docu-drama suggests and shows us a big project - building a new global society without money and oil - named Venus Project. And some people have already joined that project. They are arguing this is possible and we have that much of developed technology. The thing making it possible is technology and motivation rather than politics. I’m not worshiping this conspiracy theory and I’m not suggesting you to join the project. I don’t know the documentary movie is true. But in the world, only 1 percent of the population owns 40 percent of the planet’s wealth. In every single day, 34,000 children die for poverty and preventable diseases. And 50 percent of the population lives with only two dollars a day. In fact, any volunteering services we can do and we do have been almost nothing or limited already. I also don’t know exactly why. But, “One thing is clear. Something is very wrong.” In the middle of this ugly moment we live in history, I think you need to have your own zeitgeist or values at least. I don’t know what I have to do also, but I’m going to do something to be someone. Because I believe I need to be somebody first to make myself do something valuable. You too, don’t waste your time, turn off TV, open your eyes to happening objective facts and find something you can do. Imagine all the people sharing all the world. Yes, it’s time to wake up to be realist, having a impossible dream in your heart, twenties!"

# Define lexicons
ROMANCE_COGNATES = {
    "consequently", "therefore", "however", "nevertheless", 
    "furthermore", "moreover", "additionally", "alternatively", 
    "specifically", "particularly"
}

TRANSPORT_WORDS = {"taxi", "train", "bus", "bicycle", "subway", "metro", "bike"}

HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", 
    "seem", "appear", "somewhat", "relatively", "fairly", 
    "rather", "kind of", "sort of" 
}
# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
features = {}

def extract_romance_cognates_feature(text):
    '''
    Extract the Boolean value of romance cognates exsitence in the text we analyze, 
    since European learners have advantage with formal academic vocabulary.

    Args:
        text: Strings

    Returns:
        Boolean: True of False
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0

    features['has_romance_cognates'] = any(word in ROMANCE_COGNATES for word in words)
    return features

def extract_transport_words_feature(text):
    '''
    Extract the Boolean value of transportation words exsitence in the text we analyze, since Asian learners 
    use more public transport or bikes in their daily life.

    Args:
        text: Strings

    Returns:
        Boolean: True of False
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0

    features['TRANSPORT_WORDS'] = any(word in TRANSPORT_WORDS for word in words)
    return features

def extract_headging_words_feature(text):
    '''
    East Asian cultures tend to be more indirect in communication and may use more hedging language to soften statements, 
    while European speakers may be more direct. 
    This reflects cultural differences in politeness strategies and assertiveness.

    Args:
        text: Strings

    Returns:
        Boolean: True of False
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0

    features['HEDGING_WORDS'] = any(word in HEDGING_WORDS for word in words)
    return features

def get_article_ratio(text):
    '''
    Asian languages often lack articles → So they might underuse or misuse articles
    Args:
        text: Strings

    Returns:
        Float: Article Ratio
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0
    
    articles = sum(1 for word, tag in pos_tags if tag == 'DT' and word in ['a', 'an', 'the'])
    features['article_ratio'] = articles / total_words
    return features

def get_pronoun_density(text):
    '''
    Getting different pronoun usage patterns
    Args:
        text: Strings

    Returns:
        Float: Pronoun Ratio
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0
    
    pronouns = sum(1 for _, tag in pos_tags if tag in ['PRP', 'PRP$'])
    features['pronoun_density'] = pronouns / total_words
    return features

def get_prepositions_ratio(text):
    '''
    Getting preposition because its systems differ dramatically between language families and may cause potential difference
    Args:
        text: Strings

    Returns:
        Float: Article Ratio
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0
    
    prepositions = sum(1 for _, tag in pos_tags if tag == 'IN')
    features['preposition_ratio'] = prepositions / total_words
    return features

def get_modals_verb_ratio(text):
    '''
    Modal verb usage patterns differ
    Args:
        text: Strings

    Returns:
        Float: Modal Verb Ratio
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0

    modals = sum(1 for _, tag in pos_tags if tag == 'MD')
    features['modal_verb_ratio'] = modals / total_words
    return features

def get_adjectives_ratio(text):
    '''
    Get descriptive writing style differences
    Args:
        text: Strings

    Returns:
        Float: Adjective Ratio
    '''
    features = {}

    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # POS tagging
    pos_tags = pos_tag(words)
    
    # Safety checks
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words == 0:
        return 0
    
    adjectives = sum(1 for _, tag in pos_tags if tag in ['JJ', 'JJR', 'JJS'])
    features['adjective_ratio'] = adjectives / total_words
    return features

def create_label(l1):
    """
    Convert L1 to binary class label.
    
    Args:
        l1 (str): Native language
    
    Returns:
        str: 'European' or 'Asian', or None for other languages
    """
    if l1 in ['French', 'Spanish']:
        return 'European'
    elif l1 in ['Mandarin', 'Japanese', 'Korean']:
        return 'Asian'
    else:
        return None

def build_dataset(zip_path, train_files, dev_files, test_files):
    """
    Build train/dev/test datasets with features and labels.
    
    Args:
        zip_path: Path to lang8.zip
        train_files, dev_files, test_files: Sets of filenames for each split
    
    Returns:
        tuple: (X_train, y_train, X_dev, y_dev, X_test, y_test)
    """
    X_train, y_train = [], []
    X_dev, y_dev = [], []
    X_test, y_test = [], []

    with open(train_files) as f:
        train_set = set(line.strip() for line in f)
    
    with open(dev_files) as f:
        dev_set = set(line.strip() for line in f)
    
    with open(test_files) as f:
        test_set = set(line.strip() for line in f)
    
    # Iterate through all documents
    for l1, text, filename in iterate_documents(zip_path):
        # Skip if not European or Asian
        label = create_label(l1)
        if label is None:
            continue
        
        # Extract features
        features.update(extract_headging_words_feature(text))
        features.update(extract_romance_cognates_feature(text))
        features.update(extract_transport_words_feature(text))
        features.update(get_adjectives_ratio(text))
        features.update(get_article_ratio(text))
        features.update(get_modals_verb_ratio(text))
        features.update(get_pronoun_density(text))
        features.update(get_prepositions_ratio(text))
        
        filename  = filename.split("/")[1]
        # Add to appropriate split
        if filename in train_set:
            X_train.append(features)
            y_train.append(label)
        elif filename in dev_set:
            X_dev.append(features)
            y_dev.append(label)
        elif filename in test_set:
            X_test.append(features)
            y_test.append(label)
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

X_train, y_train = [], []
X_dev, y_dev = [], []
X_test, y_test = [], []
X_train, y_train, X_dev, y_dev, X_test, y_test = build_dataset("../data/raw/lang-8.zip","../data/dev.txt","../data/train.txt","../data/test.txt")
print(X_train)