## Dataset & paths
Create a folder `data` and add place the `raw_files` folder (containing the midi files) within. The `train.csv`, `test.csv` and `validation.csv` also have to be in the data folder.

## Tokenization and Usage
Use `create_tokenized_datasets` from `utils/generate_tokenized_sets.py` to generate tokenized tracks. Those tokenized tracks can then be read using `Dataset` from `model_old.py`.
