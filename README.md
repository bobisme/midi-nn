## Objective

Build a model capable of generating and endless piano sonata that's not obviously AI.

## Components

- `model.py` contains the model.
- `train.py` loads tokenized files, trains the model, and generates samples.
- `midi-tokenizer` CLI tool:
  - Converts midi files to tokens.
  - Converts tokens to midi files.
  - Inspects midi files.
  - Inspects token files.

## Logs

### Token Format

| Version | Changes                                                          |
| ------: | ---------------------------------------------------------------- |
|       4 | Replace event deltas with "wait" tokens.                         |
|       3 | Add Sustain Pedal and Tempo tokens. Add file header.             |
|       2 | Replace Note token + Duration param with NoteOn, NoteOff tokens. |

### Model

| Version | Changes                           |
| ------: | --------------------------------- |
|       8 | Embed size 32 -> 64. ReLU -> GELU |

### Data

- Implemented a probabilistic transposition system in the data loader.
