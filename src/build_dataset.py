from src.part_1 import iterate_documents
from src.part_2 import sentence_length_stats, text_stats
# from src.part_2_wip import extract_headging_words_feature, extract_romance_cognates_feature, ... 

def build_dataset(zip_path, train_files, dev_files, test_files):
    """
    Build train/dev/test datasets with features and labels.
    
    Args:
        zip_path: Path to lang-8.zip
        train_files, dev_files, test_files: Paths to files listing filenames for each split
    
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

    for l1, text, filename in iterate_documents(zip_path):
        # Make label (e.g., European vs Asian); skip others
        label = create_label(l1)
        if label is None:
            continue

        # Start a fresh feature dict for this document
        features = {}

        features.update(extract_headging_words_feature(text))
        features.update(extract_romance_cognates_feature(text))
        features.update(extract_transport_words_feature(text))
        features.update(get_adjectives_ratio(text))
        features.update(get_article_ratio(text))
        features.update(get_modals_verb_ratio(text))
        features.update(get_pronoun_density(text))
        features.update(get_prepositions_ratio(text))

        # === 8 features (Sentence segmentation–based, Statistical) ===
        features.update(sentence_length_stats(text, long_thresh=20))
        features.update(text_stats(text))

        # Match filename format in train/dev/test lists
        filename_only = filename.split("/")[-1]

        if filename_only in train_set:
            X_train.append(features)
            y_train.append(label)
        elif filename_only in dev_set:
            X_dev.append(features)
            y_dev.append(label)
        elif filename_only in test_set:
            X_test.append(features)
            y_test.append(label)

    return X_train, y_train, X_dev, y_dev, X_test, y_test