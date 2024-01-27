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

| Version | Changes                                              |
| ------: | ---------------------------------------------------- |
|       9 | Embed size 64 -> 192. Combined multi head attention. |
|       8 | Embed size 32 -> 64. ReLU -> GELU                    |

#### Notes

- Version 9
  - Converges to a lower loss faster than v8.
  - Uses less VRAM.
- Version 8
  - Lower loss faster than v7. Takes 2x the time for same iterations.
  - Seems to prioritize wait tokens, so lots of dead space in generated samples.

### Data Loader

| Version | Changes                                   |
| ------: | ----------------------------------------- |
|       2 | Added probabilistic transposition system. |

### To Try

- [ ] Sparse Attention.
- [ ] 384 embedding size.
- [ ] Go back to continuous wait/delay.
- [ ] Generate wait & velocity from the output of the attention block.
