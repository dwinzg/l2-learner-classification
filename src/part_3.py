import pandas as pd
import numpy as np

from pathlib import Path
from src.build_dataset import build_dataset

# Root path direction
ROOT = Path.cwd().parent
DATA = ROOT / "data"

# train, test, dev
zip_path = DATA / "raw" / "lang-8.zip"
train = DATA / "train.txt"
dev = DATA / "dev.txt"
test = DATA / "test.txt"

X_train, y_train, X_dev, y_dev, X_test, y_test = build_dataset(
    zip_path, train, dev, test
)

# -----------------------
# Vectorization
# -----------------------
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=True)

X_train_vec = vec.fit_transform(X_train)
X_dev_vec = vec.transform(X_dev)
X_test_vec = vec.transform(X_test)

# Sparse to dense arrays
X_train_array = X_train_vec.toarray()
X_dev_array = X_dev_vec.toarray()
X_test_array = X_test_vec.toarray()

# -----------------------
# Baseline model
# -----------------------
from sklearn.tree import DecisionTreeClassifier

baseline = DecisionTreeClassifier(random_state=521, max_depth=3)
baseline.fit(X_train_array, y_train)
print(f"baseline accuracy = {baseline.score(X_dev_array, y_dev)}")

# -----------------------
# Feature ablation
# -----------------------
def feature_ablation(vec, index, X_train_array, y_train, X_dev_array, y_dev):
    baseline = DecisionTreeClassifier(random_state=521, max_depth=4)
    baseline.fit(X_train_array, y_train)
    base_score = baseline.score(X_dev_array, y_dev)

    n_features = X_train_array.shape[1]
    keep_cols = np.delete(np.arange(n_features), index)

    X_train = X_train_array[:, keep_cols]
    X_dev = X_dev_array[:, keep_cols]

    model = DecisionTreeClassifier(random_state=521, max_depth=4)
    model.fit(X_train, y_train)

    accuracy = model.score(X_dev, y_dev)
    delta = base_score - accuracy

    return index, vec.feature_names_[index], accuracy * 100, delta * 100


result = []
for index, feature in enumerate(vec.feature_names_):
    result.append(
        feature_ablation(
            vec, index, X_train_array, y_train, X_dev_array, y_dev
        )
    )

result = sorted(result, key=lambda x: x[3])

ablation_features = []
for index, feature, score, delta in result:
    if delta < 0:
        ablation_features.append(index)

# -----------------------
# Retrain with selected features
# -----------------------
test_tree = DecisionTreeClassifier(max_depth=4, random_state=521)

n_features = X_train_array.shape[1]
keep_cols = np.delete(np.arange(n_features), ablation_features)

test_tree.fit(X_train_array[:, keep_cols], y_train)

print(test_tree.score(X_dev_array[:, keep_cols], y_dev))
