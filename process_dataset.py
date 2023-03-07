
import random
import re
from collections import Counter

import pandas as pd


def main():

    """
    class:
        0 -> hate speech
        1 -> offensive language
        2 -> neither

    tweet_id,class,tweet_body,split
    0,2,"!!!...",training
    ...

    Class Balance:
        0: 1400
        1: 2800
        2: 2800
        total: 7000

    Train/Val/Test Split:
        60/20/20
    """
    """
    class_counter = Counter()
    verycertain_class_counter = Counter()
    threshold = 3
    certain_class_counter = Counter()
    dif1_class_counter = Counter()
    dif2_class_counter = Counter()
    count_counter = Counter(raw_data["count"].to_list())
    print("count_counter", count_counter)

    for index, row in raw_data.iterrows():
        class_counter[row["class"]] += 1
        if (row["class"] == 0 and row["hate_speech"] == row["count"] and row["count"] > threshold) or (row["class"] == 1 and row["offensive_language"] == row["count"] and row["count"] > threshold) or (row["class"] == 2 and row["neither"] == row["count"] and row["count"] > threshold):
            verycertain_class_counter[row['class']] += 1
        elif (row["class"] == 0 and row["hate_speech"] == row["count"]) or (row["class"] == 1 and row["offensive_language"] == row["count"]) or (row["class"] == 2 and row["neither"] == row["count"]):
            certain_class_counter[row['class']] += 1
        elif (row["class"] == 0 and row["hate_speech"] == (row["count"] - 1)) or (row["class"] == 1 and row["offensive_language"] == (row["count"] - 1)) or (row["class"] == 2 and row["neither"] == (row["count"] - 1)):
            dif1_class_counter[row['class']] += 1
        elif (row["class"] == 0 and row["hate_speech"] == (row["count"] - 2)) or (row["class"] == 1 and row["offensive_language"] == (row["count"] - 2)) or (row["class"] == 2 and row["neither"] == (row["count"] - 2)):
            dif2_class_counter[row['class']] += 1

    print("class_counter", class_counter)
    print("verycertain_class_counter", verycertain_class_counter)
    print("certain_class_counter", certain_class_counter)
    print("dif1_class_counter", dif1_class_counter)
    print("dif2_class_counter", dif2_class_counter)
    """

    random.seed(42)

    raw_data = pd.read_csv("original_dataset.csv")

    # Select data. Highest quality first, then fill the rest with less good data
    hate_speech_subset = raw_data[(raw_data["class"] == 0) & (raw_data["count"] - raw_data["hate_speech"] < 2)]  # 1397
    # .sample() samples a random number of rows from a dataframe, .sample(frac=1) shuffles the dataset (use sample instead of .head())
    hate_speech_subset = pd.concat([hate_speech_subset, raw_data[(raw_data["class"] == 0) & (raw_data["count"] - raw_data["hate_speech"] == 2)].sample(1400 - hate_speech_subset.shape[0])]).sample(frac=1)
    # Create list of split strings of the right size and distribution, then shuffle it
    hate_speech_split_list = ["train"] * int(hate_speech_subset.shape[0] * 0.6) + ["validation"] * int(hate_speech_subset.shape[0] * 0.2) + ["test"] * int(hate_speech_subset.shape[0] * 0.2)
    random.shuffle(hate_speech_split_list)
    # Add split list to dataframe as a column with title "split" (Warning: list has to be the same size as the dataframe)
    hate_speech_subset["split"] = hate_speech_split_list
    # Check that no data was accidentally selected twice
    if hate_speech_subset.index.has_duplicates:
        raise

    offensive_subset = raw_data[(raw_data["class"] == 1) & (raw_data["count"] - raw_data["offensive_language"] < 2) & (raw_data["count"] > 3)]  # 1491
    offensive_subset = pd.concat([offensive_subset, raw_data[(raw_data["class"] == 1) & (raw_data["count"] - raw_data["offensive_language"] == 0) & (raw_data["count"] == 3)].sample(2800 - offensive_subset.shape[0])])
    offensive_split_list = ["train"] * int(offensive_subset.shape[0] * 0.6) + ["validation"] * int(offensive_subset.shape[0] * 0.2) + ["test"] * int(offensive_subset.shape[0] * 0.2)
    random.shuffle(offensive_split_list)
    offensive_subset["split"] = offensive_split_list
    if offensive_subset.index.has_duplicates:
        raise

    neither_subset = raw_data[(raw_data["class"] == 2) & (raw_data["count"] - raw_data["neither"] < 2) & (raw_data["count"] > 3)]  # 195
    neither_subset = pd.concat([neither_subset, raw_data[(raw_data["class"] == 2) & (raw_data["count"] - raw_data["neither"] == 0) & (raw_data["count"] == 3)].sample(2800 - neither_subset.shape[0])])
    neither_split_list = ["train"] * int(neither_subset.shape[0] * 0.6) + ["validation"] * int(neither_subset.shape[0] * 0.2) + ["test"] * int(neither_subset.shape[0] * 0.2)
    random.shuffle(neither_split_list)
    neither_subset["split"] = neither_split_list
    if neither_subset.index.has_duplicates:
        raise

    new_data = pd.concat([hate_speech_subset, offensive_subset, neither_subset]).sort_values(by="id").reset_index(drop=True).drop(columns=["hate_speech", "offensive_language", "neither", "count"]).rename(columns={"tweet": "tweet_body", "id": "tweet_id"})
    print(new_data)
    print(Counter(new_data["split"]))

    new_data["tweet_body"] = new_data["tweet_body"].apply(lambda s: re.sub("&#(\\d+);", r" &#\1; ", s.replace('"', ' ').replace('&lt;', ' &lt; ').replace('&gt;', ' &gt; ')))

    new_data.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    main()
