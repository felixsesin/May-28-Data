from diffusion import State
from helpers import getGrid
import time

vals = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0]

lines = []

i = 1

for ensemble in ('verysmall.json', 'small.json', 'large.jsonl'):
    if ensemble != 'verysmall.json': continue
    start = time.time()

    for a in vals:
        if a != 0.2: continue
        for w in vals:
            if w != 0.5: continue
            state = State(alpha = a,
                        omega = w,
                        precinct_map = getGrid(4,4),
                        file = ensemble)

            state.saveInfo()
            state.getLambdaLambda(1,2,save=True)
            state.getSpectralDecay(save=True)

            del state

            print(f"{i} / 507 runs complete")
            i += 1

    line = f"The file '{ensemble}' took {time.time()-start} seconds!!!"
    lines.append(line)

    del start, line

    
with open('you_should_read_meeee.txt', 'w') as f:
    f.write('\n'.join(lines))