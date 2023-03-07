
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

    dataset = pd.read_csv("dataset.csv")

    dataset_size = dataset.shape[0]

    print(f"Total number of items: {dataset_size}")
    print(f"Number of Hate Speech items: {dataset[dataset['class'] == 0].shape[0]}")
    print(f"Number of Offensive Language items: {dataset[dataset['class'] == 1].shape[0]}")
    print(f"Number of Neither items: {dataset[dataset['class'] == 2].shape[0]}")

    print(f"Size of train split: {dataset[dataset['split'] == 'train'].shape[0]} ({100 * (dataset[dataset['split'] == 'train'].shape[0] / dataset_size)}%)")
    print(f"Size of validation split: {dataset[dataset['split'] == 'validation'].shape[0]} ({100 * (dataset[dataset['split'] == 'validation'].shape[0] / dataset_size)}%)")
    print(f"Size of test split: {dataset[dataset['split'] == 'test'].shape[0]} ({100 * (dataset[dataset['split'] == 'test'].shape[0] / dataset_size)}%)")

    print("Label distribution of train split:")
    print(f"Hate Speech {dataset[(dataset['class'] == 0) & (dataset['split'] == 'train')].shape[0]}")
    print(f"Offensive Language {dataset[(dataset['class'] == 1) & (dataset['split'] == 'train')].shape[0]}")
    print(f"Neither {dataset[(dataset['class'] == 2) & (dataset['split'] == 'train')].shape[0]}")

    print("Label distribution of validation split:")
    print(f"Hate Speech {dataset[(dataset['class'] == 0) & (dataset['split'] == 'validation')].shape[0]}")
    print(f"Offensive Language {dataset[(dataset['class'] == 1) & (dataset['split'] == 'validation')].shape[0]}")
    print(f"Neither {dataset[(dataset['class'] == 2) & (dataset['split'] == 'validation')].shape[0]}")

    print("Label distribution of test split:")
    print(f"Hate Speech {dataset[(dataset['class'] == 0) & (dataset['split'] == 'test')].shape[0]}")
    print(f"Offensive Language {dataset[(dataset['class'] == 1) & (dataset['split'] == 'test')].shape[0]}")
    print(f"Neither {dataset[(dataset['class'] == 2) & (dataset['split'] == 'test')].shape[0]}")


if __name__ == "__main__":
    main()
