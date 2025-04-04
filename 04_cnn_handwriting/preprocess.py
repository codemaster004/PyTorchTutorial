import os

import pandas as pd


def create_dataset_csv():
	# For the dataset
	dataset = []
	
	# Data location
	source_dir = "./data/"
	
	author_dirs = os.listdir(source_dir)
	for sub_dir in author_dirs:
		# Make sure it is a dir
		if not os.path.isdir(os.path.join(source_dir, sub_dir)):
			continue
		# File with position and labels of words
		info_file = os.path.join(source_dir, sub_dir, "word_places.txt")
		with open(info_file, 'r', encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip()  # Remove leading, trailing whitespace
				# Some lines can have comments
				if line[0] == "%":
					continue
				# Read the values in each row
				after_split = line.split(" ")
				path = after_split[0]
				path = path.replace('"', '')
				path = os.path.join(source_dir, sub_dir, path)
				y1, x1, y2, x2 = after_split[-4], after_split[-3], after_split[-2], after_split[-1]
				label = " ".join(after_split[1:-4])
				print(label)
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				# Add to dataset
				dataset.append({"path": path, "label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
	# Save the file
	pd.DataFrame(dataset).to_csv("./data/dataset.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
	create_dataset_csv()
