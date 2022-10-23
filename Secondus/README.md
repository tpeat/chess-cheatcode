# Chess Board / Pieces CV

Based on [iTurk](https://github.com/TrentConley/iTurk) developed by [@TrentConley](https://github.com/TrentConley)
And [Chess Ray Visison](https://github.com/samryan18/chess-ray-vision) by [@SamRyan18](https://github.com/samryan18)

Thank you for the inspiration

Goal is to divid

## Dataset
Can be found [here](https://github.com/samryan18/chess-dataset.git)

## Usage
python3 -m pip install requirements.txt

NOTE: 
- tensorflow is nasty on M1, follow the [apple developer guidelines](https://developer.apple.com/forums/thread/686926)
- stockfish was a little tricky, `python -m pip install stockfish` did the trick tho

To run the chess recognition, `run secondus.py`

## Preprocessing

We will take a photo of a larger board and will have to like crop the image to somehting smaller before we run our trained CNN over what we have to generate the FEN 

## TODO

Secondus.py has a formatting issue, going to require either:
- refitting the model (which trent saids takes 15 hours)
- reworking the way the model works

Reworked everything to fit in `main.py`:
- getting a batch size mismatch in the forward layer of all nets

