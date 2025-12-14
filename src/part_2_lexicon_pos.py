import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from collections import Counter
import string

# Define lexicons
Asian_words = {
    "china", "chinese", "korea", "korean", 
    "japan", "japanese", "tokyo", "beijing", 
    "shanghai", "seoul"
}

European_words = {
    "Spain", "Spanish", "French", "France", "Paris",
    "Madrid", "Barcelona", "Lyon", "Merci"
}

Religion_words = {
    "catholic", "christ", "god", "church",
    "pray", "angel", "holy", "spirit", "devil"
}

#Define Noun tags
NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}


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
    total_words = len(words)
    if total_words == 0:
        features['asian_top_word_match'] = False
        features['Religious_Feature'] = False
        return features
    
    pos_tagged_words = pos_tag(words)
    nouns = [word for word, tag in pos_tagged_words if tag in NOUN_TAGS]

    # Calculate Word Frequencies
    word_counts = Counter(nouns)
    
    # Get the Top 3 Most Frequent Words
    top_3_list = word_counts.most_common(3)
    
    # Extract just the word strings
    top_3_words = [item[0] for item in top_3_list]

    match_count = 0
    
    for word in top_3_words:
        # Check if the word is in the Asian_words
        if word in Asian_words:
            match_count += 1
            
    # The feature is True if 2 or more words matched the lexicon
    features['asian_top_word_match'] = (match_count >= 2)
    features['Religious_Feature'] = any(word in Religion_words  for word in words)
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