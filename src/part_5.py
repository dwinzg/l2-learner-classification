import pandas as pd
import altair as alt
import altair_ally as aly
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV

from scipy.stats import loguniform, randint
from pathlib import Path

# ---------- Image output directory ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "images" / "part_5"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# DataFrames for EDA
# -----------------------
X_train = pd.DataFrame(X_train)
X_dev = pd.DataFrame(X_dev)
X_test = pd.DataFrame(X_test)

train = X_train.copy()
train["target"] = y_train

cat_cols = train.select_dtypes(include=["bool", "object"]).columns.tolist()

# -----------------------
# EDA
# -----------------------
aly.alt.data_transformers.enable("vegafusion")

dist_chart = aly.dist(train, color="target").resolve_scale(y="independent")
dist_chart.save(IMAGE_DIR / "dist_target.png")

cat_train = train[cat_cols].melt(
    id_vars="target", var_name="variable", value_name="value"
)

cat_chart = alt.Chart(cat_train).mark_bar().encode(
    x=alt.X("value:N").title("Category"),
    y=alt.Y("count()").title("Count"),
    color="target",
).properties(
    width=150,
    height=150,
).facet(
    "variable:N",
    columns=3,
)
cat_chart.save(IMAGE_DIR / "categorical_distributions.png")

corr_chart = aly.corr(train)
corr_chart.save(IMAGE_DIR / "correlation.png")

# -----------------------
# Tree visualization
# -----------------------
plt.figure(figsize=(14, 8))
plot_tree(
    test_tree,
    feature_names=vec.feature_names_,
    class_names=["Asian", "European"],
    filled=True,
    impurity=False,
    fontsize=10,
)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "decision_tree.png", dpi=200)
plt.close()

# -----------------------
# Grid Search
# -----------------------
search_grid = {
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": [None, "sqrt", "log2"],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

tree = DecisionTreeClassifier(random_state=521)

grid_search = GridSearchCV(
    estimator=tree,
    param_grid=search_grid,
    scoring="accuracy",
)

grid_search.fit(X_train_array, y_train)

best_tree = grid_search.best_estimator_

# -----------------------
# RFECV
# -----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=521)

rfecv = RFECV(
    estimator=best_tree,
    n_jobs=-1,
    verbose=1,
    cv=cv,
    scoring="accuracy",
)

rfecv.fit(X_train_array, y_train)

selected_features = [
    f for f, keep in zip(vec.feature_names_, rfecv.get_support()) if keep
]

print(
    "Dev prediction score with hyperparameter optimization and feature selection:",
    rfecv.score(X_dev_array, y_dev),
)

print(
    "Test prediction score with hyperparameter optimization and feature selection:",
    rfecv.score(X_test_array, y_test),
)
