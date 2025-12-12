import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from collections import Counter
import string

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
    "rather", "kind", "sort" 
}
# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
features = {}

def extract_lexicon_features(text):
    '''
    Extract the Boolean value of romance cognates exsitence in the text we analyze, 
    since European learners have advantage with formal academic vocabulary.

    Extract the Boolean value of transportation words exsitence in the text we analyze, since Asian learners 
    use more public transport or bikes in their daily life.

    East Asian cultures tend to be more indirect in communication and may use more hedging language to soften statements, 
    while European speakers may be more direct. 
    This reflects cultural differences in politeness strategies and assertiveness.
    Args:
        text: Strings

    Returns:
        tuple: (Boolean, Boolean, Boolean)
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
    features['HEDGING_WORDS'] = any(word in HEDGING_WORDS for word in words)
    features['TRANSPORT_WORDS'] = any(word in TRANSPORT_WORDS for word in words)
    return features

def get_POS_rato_features(text):
    '''
    Asian languages often lack articles → So they might underuse or misuse articles.
    
    Getting different pronoun usage patterns
    
    Getting preposition because its systems differ dramatically between language families and may cause potential difference

    Modal verb usage patterns differ

    Get descriptive writing style differences

    Args:
        text: Strings

    Returns:
        tuple: (Float, Float, Float, Float, Float)
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

    pronouns = sum(1 for _, tag in pos_tags if tag in ['PRP', 'PRP$'])
    features['pronoun_density'] = pronouns / total_words
    
    prepositions = sum(1 for _, tag in pos_tags if tag == 'IN')
    features['preposition_ratio'] = prepositions / total_words

    modals = sum(1 for _, tag in pos_tags if tag == 'MD')
    features['modal_verb_ratio'] = modals / total_words

    adjectives = sum(1 for _, tag in pos_tags if tag in ['JJ', 'JJR', 'JJS'])
    features['adjective_ratio'] = adjectives / total_words

    return features
