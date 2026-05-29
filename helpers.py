from pathlib import Path
import pickle

"""

    generates graph for a grid of width and height
    returns in form dict[int, list[int]]
    1-indexed
"""

def getGrid(width: int, height: int) -> dict[int, list[int]]:

    ret = {}

    for i in range(width * height):

        ind = i + 1
        row, col = i // width, i % width

        connected = []

        # Up
        if row > 0: connected.append(ind - width) # up
        if row < height - 1: connected.append(ind + width) # down
        if col > 0: connected.append(ind - 1) # lefts
        if col < width - 1: connected.append(ind + 1) # right

        ret[ind] = sorted(connected)

    return ret



"""
    loads a dataset from redistricting2/results
    returns in dictionary form with keys:

        'file',       'precincts', 'districts', 'omega',
        'alpha',      'nodes',     'len',       'vertical',
        'horizontal', 'diffusion', 'eigenset',  'time']
"""

def loadResults(file: str, alpha: float, omega: float) -> dict:

    if file in ('verysmall', 'verysmall.json'):
        file = 'verysmall.json'
        subfolder = 'verysmall'

    elif file in ('small', 'small.json'):
        file = 'small.json'
        subfolder = 'small'

    elif file in ('large', 'large.jsonl'):
        file = 'large.jsonl'
        subfolder = 'large'

    else:
        raise LookupError(
            f"File not found. Try:\n" +
            f"'verysmall'\n" +
            f"'small'\n" +
            f"'large'"
        )


    path = Path(__file__).parent / "results" / subfolder
    name = f"{subfolder}_alpha_{alpha}_omega_{omega}.pkl"
    data_dir = path / name

    loaded = open(data_dir, "rb")

    return pickle.load(loaded)

