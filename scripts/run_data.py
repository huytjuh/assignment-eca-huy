from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from configs.global_config import get_data_config

from src.emailurgency.pipelines.dataloader import DataLoader
from src.emailurgency.pipelines.deduplicate import Deduplicate
from src.emailurgency.pipelines.datasetbuilder import DatasetBuilder

def main() -> None:
    data_config = get_data_config()
    dataloader = DataLoader()
    data = dataloader.run()

    # deduplicate
    deduplicate = Deduplicate()
    data = deduplicate.remove_duplicates(data)

    # build dataset
    dataset_builder = DatasetBuilder()
    df = dataset_builder.transform(data)

    # manual gold label
    gold_labels = pd.read_parquet(data_config.gold_label_path)
    df = df.merge(gold_labels, on='message_id', how='left')

    # train test split
    df_nan = df[df['gold_labels'].isna()]
    df_labeled = df[df['gold_labels'].isin([0,1])]

    train_parts = []
    test_parts = []
    for subset in (df_nan, df_labeled):
        thread_ids = subset['thread_id'].astype(str).unique().tolist()
        if not thread_ids:
            continue

        train_idx, test_idx = train_test_split(thread_ids, test_size=data_config.test_size, random_state=data_config.random_seed)
        train_parts.append(subset[subset['thread_id'].isin(train_idx)])
        test_parts.append(subset[subset['thread_id'].isin(test_idx)])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    train.to_parquet('data/train.parquet', engine='pyarrow', index=False)
    test.to_parquet('data/test.parquet', engine='pyarrow', index=False)

if __name__ == '__main__':
    main()
