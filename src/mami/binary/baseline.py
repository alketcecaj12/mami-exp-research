from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_baselines(train, seed):
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_lr": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    }
    for model in models.values():
        model.fit(train.text, train.label)
    return models
