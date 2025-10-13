from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
	data_path = Path(r"C:\Users\User\Desktop\martins\labels_cleaned.csv")
	output_dir = Path(r"C:\Users\User\Desktop\martins\plots")
	output_dir.mkdir(parents=True, exist_ok=True)

	labels = pd.read_csv(data_path)
	target_cols = [col for col in labels.columns if col.startswith("T")]

	if not target_cols:
		raise ValueError("No columns starting with 'T' found in the dataset.")

	for col in target_cols:
		plt.figure(figsize=(6, 4))
		plt.hist(labels[col].dropna(), bins=30, color="#4C72B0", edgecolor="black")
		plt.title(col)
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		plt.tight_layout()

		output_path = output_dir / f"{col}.png"
		plt.savefig(output_path, dpi=300)
		plt.close()


if __name__ == "__main__":
	main()