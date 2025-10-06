import argparse
from pathlib import Path
from typing import List

import torch

from .data_utils import make_dataloaders_by, summarize_dataset


def main():
	parser = argparse.ArgumentParser(description="Build dataloaders for OpenFace AU sequences with optional padding")
	parser.add_argument("--root", required=True, help="Directory containing OpenFace CSV files")
	parser.add_argument("--field", choices=["identifier", "type", "person"], default="identifier")
	parser.add_argument("--train_values", nargs="+", required=True, help="Values for the chosen field to include in TRAIN split")
	parser.add_argument("--val_values", nargs="*", help="Values for the chosen field to include in VAL split")
	parser.add_argument("--test_values", nargs="*", help="Values for the chosen field to include in TEST split")
	parser.add_argument("--include_types", nargs="*", help="Filter to these types before splitting")
	parser.add_argument("--include_persons", nargs="*", help="Filter to these persons before splitting")
	parser.add_argument("--include_identifiers", nargs="*", help="Filter to these identifiers before splitting")
	parser.add_argument("--exclude_identifiers", nargs="*", help="Exclude these identifiers before splitting")
	parser.add_argument("--au_prefer", choices=["r", "c"], default="r", help="Prefer AU*_r (intensity) or AU*_c (presence)")
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--no_shuffle_train", action="store_true", help="Disable shuffling in train loader")
	parser.add_argument("--keep_na", action="store_true", help="Keep rows with NA values")
	parser.add_argument("--no_pad", action="store_true", help="Do not pad sequences in the collate function")
	parser.add_argument("--print_summary", action="store_true", help="Print available identifiers/types/persons and exit")
	args = parser.parse_args()

	root = Path(args.root)

	if args.print_summary:
		summary = summarize_dataset(root)
		print("Identifiers:", summary["identifier"]) 
		print("Types:", summary["type"]) 
		print("Persons:", summary["person"]) 
		return

	loaders = make_dataloaders_by(
		root=root,
		field=args.field,
		train_values=args.train_values,
		val_values=args.val_values if args.val_values else None,
		test_values=args.test_values if args.test_values else None,
		include_types=args.include_types,
		include_persons=args.include_persons,
		include_identifiers=args.include_identifiers,
		exclude_identifiers=args.exclude_identifiers,
		au_prefer=args.au_prefer,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		shuffle_train=not args.no_shuffle_train,
		drop_na=not args.keep_na,
		pad=not args.no_pad,
	)

	# Quick smoke test: iterate one batch from each split
	for split, loader in loaders.items():
		print(f"Split {split}: {len(loader.dataset)} files")
		for batch in loader:
			if isinstance(batch["x"], list):
				lens = [t.shape for t in batch["x"]]
				print(f"  Batch lens: {lens}; meta_count={len(batch['meta'])}")
			else:
				x = batch["x"]
				print(f"  Batch x shape: {tuple(x.shape)}; lengths={batch['lengths'].tolist()}")
			break


if __name__ == "__main__":
	main()


# Examples:
# python -m source.scripts_face.train_model --root /home/data_shares/genface/data/MentalHealth/msb/OpenFace_Output_MSB/ --print_summary
# python -m source.scripts_face.train_model --root /home/data_shares/genface/data/MentalHealth/msb/OpenFace_Output_MSB/ --field type --train_values Wunder Personal --include_persons Pr --no_pad