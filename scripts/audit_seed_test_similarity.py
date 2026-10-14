
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

seed = pd.read_csv("data/indian_merchants_train_seed.csv")
base = pd.read_csv("data/mitulshah_split.csv")

test = base.loc[
    base["split"] == "test", "transaction_description"
].fillna("").astype(str)

normalize = lambda x: " ".join(str(x).lower().split())

seed_text = seed["transaction_description"].fillna("").map(normalize)
test_text = test.map(normalize)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1
)

combined = pd.concat([seed_text, test_text], ignore_index=True)
X = vectorizer.fit_transform(combined)

X_seed = X[:len(seed_text)]
X_test = X[len(seed_text):]

nn = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)
nn.fit(X_test)

distances, indices = nn.kneighbors(X_seed)

results = pd.DataFrame({
    "seed_description": seed["transaction_description"].astype(str),
    "nearest_test_description": test.iloc[indices[:, 0]].to_numpy(),
    "similarity": 1 - distances[:, 0]
})

print("Seed rows:", len(seed))
print("Main test rows:", len(test))
print("Similarity >= 0.90:", (results["similarity"] >= 0.90).sum())
print("Similarity >= 0.80:", (results["similarity"] >= 0.80).sum())

print("\nTop 20 closest matches:")
print(
    results.sort_values("similarity", ascending=False)
    .head(20)
    .to_string(index=False)
)