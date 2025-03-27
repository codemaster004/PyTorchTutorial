import kagglehub
import shutil
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def download():
	# Make sure directory exists
	os.makedirs("./data", exist_ok=True)
	
	# Download latest version of the file
	path = kagglehub.dataset_download("rahmasleam/flowers-dataset")
	print(f"Downloaded to: {path}")
	shutil.copytree(path, "./data", dirs_exist_ok=True)


# As importing the images is hard, first create a file with label, image path
def pre_process():
	dataset = []  # later to convert to pd.DataFrame
	path = Path('./data')  # using pathlib, chat suggestion
	
	for item in path.rglob('*'):  # Iterate over all files, even those in sub-dirs
		if item.is_file() and item.suffix in ['.jpg', '.png', '.jpeg']:  # Only accept images
			dataset.append({"label": item.parts[-2], "path": item})  # Add (label, path) to dataset
	
	return pd.DataFrame(dataset)  # return pd.DataFrame


def split_dataset(_df):
	# Split into values, labels to enabling grouping later
	X = _df.drop(columns=['label'])  # Features
	y = _df['label']  # Labels
	
	# Split the data, grouped by labels
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.1, stratify=y
	)
	
	# Re-assemble the train and test DataFrames
	train_df = X_train.copy()
	train_df['label'] = y_train
	test_df = X_test.copy()
	test_df['label'] = y_test
	# Return both dataframes
	return train_df, test_df


if __name__ == '__main__':
	download()  # Download raw dataset
	df = pre_process()  # generate (label, path) file
	train, test = split_dataset(df)  # Split into train, test datasets
	# Save to files
	train.to_csv("./data/train.csv", index=False)
	test.to_csv("./data/test.csv", index=False)
