import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

print("Loading data...")
train_df = pd.read_csv("./input/en_train.csv")
test_df = pd.read_csv("./input/en_test_2.csv")

# Filter out cases where normalization isn't needed (before == after)
train_df = train_df[train_df["before"] != train_df["after"]]

# Build lookup tables for each class
class_mappings = defaultdict(dict)
print("Building class mappings...")
for class_name in train_df["class"].unique():
    class_data = train_df[train_df["class"] == class_name]
    # Get most common after value for each before value
    mappings = class_data.groupby("before")["after"].agg(lambda x: x.mode()[0])
    class_mappings[class_name] = mappings.to_dict()


def normalize_token(before, class_name):
    """Normalize token using lookup tables"""
    if class_name in class_mappings:
        return class_mappings[class_name].get(before, before)
    return before


# Validation split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
val_df["predicted"] = val_df.apply(
    lambda row: normalize_token(row["before"], row["class"]), axis=1
)
accuracy = (val_df["predicted"] == val_df["after"]).mean()
print(f"Validation Metric: {accuracy:.6f}")

# Process test data - initialize with original tokens
test_df["after"] = test_df["before"]

# Apply normalization for known classes
for class_name in class_mappings:
    class_tokens = set(class_mappings[class_name].keys())
    mask = test_df["before"].isin(class_tokens)
    test_df.loc[mask, "after"] = test_df.loc[mask, "before"].map(
        lambda x: normalize_token(x, class_name)
    )

# Generate submission
submission_df = pd.DataFrame(
    {
        "id": test_df["sentence_id"].astype(str)
        + "_"
        + test_df["token_id"].astype(str),
        "after": test_df["after"],
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)
print("Submission file saved successfully.")
