# Music Genre Classification for MIDI data
## Requirements
This project uses Python 3.12. The required packages can be installed from `requirements.txt`.

## Structure of the repository
### `models/GoePT/`
This folder contains the main code. 
- `train.py` is used for training. Learning rate, batch size, ... can be configured trough command line arguments (see `run.sh` for an example).
- `model.py` and `layers.py` contain the main code of the transformer
- `dataset.py` contains the relevant code for loading and handling the tokenized midi data

### `models/tokenizers/`
- `midi_tokenizer.json` contains the vocabulary of the tokenized midi files
  
### `utils/`
- `prepare_dataset.ipynb` was used to clean up the dataset and split it into train, validation and test sets
- `generate_midiGenre_tokenizer.py` was used to generate the tokenizer and tokenize the dataset
- `decode_tokenizer_midi.py` was used to ensure correct encoding
- `test_model.ipynb` was used to test and plot the accuracy of the trained model
- 

### `data/`
This folder contains all relevant data for training, including
- raw midi files (`data/raw_files`)
- hashes of the tracks of test, validation and train sets (e.g. `train.csv`)
- tokenized tracks (`data/tokenized`)

### `checkpoints/`
Includes checkpoints that can be used to run and test the model.
E.g. `class_rock_pop_80percent.json` was trained on classical, rock and pop genres and had a validation accuracy of approximatly 80%.
