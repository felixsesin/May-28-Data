import ast
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import pickle
import time



"""
For the sake of diffusion, all precincts must be assigned to a district.
Precincts in districts would never be able to diffuse (V or H) to a precinct
not in a district, and so what happens with those precincts would be irrelevant
"""


class State:

    def __init__(self,
                 alpha: float,
                 omega: float,
                 precinct_map: dict[int, list[int]],
                 districts: dict[str, list[int]] = {},
                 file: str = 'NONE'
                 ):

        self.file = file

        if districts != {} and file == 'NONE': pass
        elif districts == {} and file != 'NONE': districts = self.unpack(file)
        else: raise ValueError('Either give districts or give file to parse')

        good = self.validate(precinct_map, districts)
        if not good:
            raise ValueError('Improper precincts and districts.')

        self.precincts = precinct_map
        self.districts = districts

        self.alpha = alpha
        self.omega = omega

        start = time.time()

        self.nodes = self.getNodes()
        self.n = len(self.nodes)

        self.vertical = self.getVertical()
        self.horizontal = self.getHorizontal()

        self.diffusion = self.getDiffusion()

        self.eigenset = self.getEigenset()

        end = time.time()
        self.time = end - start

        self.info = {
                'file': self.file,
                'precincts': self.precincts,
                'districts': self.districts,
                
                'omega': self.omega,
                'alpha': self.alpha,

                'nodes': self.nodes,
                'len': self.n,

                'vertical': self.vertical,
                'horizontal': self.horizontal,
                'diffusion': self.diffusion,

                'eigenset': self.eigenset,

                'time': self.time
        }

    def unpack(self, file: str) -> dict[str, list[int]]:

        districts = {}
        d = 0

        path = Path(__file__).parent / "ensembles" / file

        with open(path) as f:
            for i, line in enumerate(f):

                if i < 3:
                    continue

                districting = ast.literal_eval(line)["districting"]

                groups = {1: [], 2: [], 3: [], 4: [], 5: []}

                for pair in districting:
                    for key, val in pair.items():

                        x, y = ast.literal_eval(ast.literal_eval(key)[0])
                        ind = x + 4*y + 1

                        groups[val].append(ind)

                for k in range(1, 6):
                    districts[str(d + k)] = sorted(groups[k])

                d += 5

        return districts

    def validate(self, precincts: dict[int, list[int]],
                districts: dict[str, list[int]]) -> bool:

        n = len(precincts)

        i = 1

        for precinct, connected in precincts.items():

            # precincts must be 1-indexed and sorted
            if (not isinstance(precinct, int) or
                precinct != i):
                return False

            # precincts that this precinct is connected to must be given as list
            if not isinstance(connected, list): return False

            # check that connected precincts are properly named
            # check that all connections are symmetric
            for j in connected:
                if j-1 not in range(n): return False
                if i not in precincts[j]: return False

            # sort list of connected precincts just in case
            precincts[precinct] = sorted(connected)

            i += 1


        seenDistricts = []

        for district, comps in districts.items():

            # store seen district names so as to avoid duplicates
            if (not isinstance(district, str) or
                district in seenDistricts): return False
            seenDistricts.append(district)

            # precincts in district must be given as list of unique precincts
            seenPrecincts = []
            if not isinstance(comps, list): return False
            for precinct in comps:
                if precinct not in precincts.keys(): return False
                if precinct in seenPrecincts: return False
                seenPrecincts.append(precinct)

            districts[district] = sorted(comps)

        return True

    def getNodes(self) -> list[tuple[str, int]]:
        
        districts = self.districts
        nodes = []

        for district, precincts in districts.items():
         
            for precinct in precincts:
                nodes.append((district, precinct))
        
        return nodes

    def normalize(self, matrix: NDArray) -> NDArray:

        rows = []

        for row in matrix:
            tot = sum(row)
            nonzero = (tot != 0.0)
            
            if nonzero:
                newRow = [float(x)/tot for x in row]
            else:
                newRow = [float(x) for x in row]
            
            rows.append(newRow)

        return np.array(rows)

    def canVerticallyDiffuse(self, i: int, j: int) -> bool:

        node1, node2 = self.nodes[i], self.nodes[j]

        sameDistrict = (node1[0] == node2[0])
        bordering = node2[1] in self.precincts[node1[1]]

        return sameDistrict and bordering
    
    def canHorizontallyDiffuse(self, i: int, j: int) -> bool:

        node1, node2 = self.nodes[i], self.nodes[j]

        diffDistrict = (node1[0] != node2[0])
        samePrecinct = (node1[1] == node2[1])

        return diffDistrict and samePrecinct
 
    def hamming(self, i: int, j: int) -> int:

        districts = self.districts

        dist1 = districts[self.nodes[i][0]]
        dist2 = districts[self.nodes[j][0]]

        diff = []
        for precinct in dist1:
            if precinct not in dist2: diff.append(precinct)
        for precinct in dist2:
            if precinct not in dist1: diff.append(precinct)

        return len(diff)

    def getVertical(self) -> NDArray:
    
        n = self.n

        rows = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):

            for j in range(i+1, n):

                if self.canVerticallyDiffuse(i,j):
                    rows[i][j] = 1.0
                    rows[j][i] = 1.0

        vertical = np.array(rows, dtype=float)
        return self.normalize(vertical)
    
    def getHorizontal(self) -> NDArray:
    
        n = self.n

        rows = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):

            for j in range(i+1, n):

                if self.canHorizontallyDiffuse(i,j):
                    dist = self.hamming(i,j)
                    kernel = np.exp(-self.alpha * dist)
                    rows[i][j] = float(kernel)
                    rows[j][i] = float(kernel)

        horizontal = np.array(rows, dtype=float)
        return self.normalize(horizontal)
    
    def getDiffusion(self) -> NDArray:

        vertical = self.vertical
        horizontal = self.horizontal
        omega = self.omega

        diffusion = omega*vertical + (1-omega)*horizontal
        return self.normalize(diffusion)
    
    def getEigenset(self) -> list[tuple[float, list[int]]]:

        eigenset = np.linalg.eig(self.diffusion)
        lambdas = [float(lam) for lam in eigenset[0]]
        n = len(lambdas)
        vecs = [eigenset[1][:, i] for i in range(n)]

        eigenlist = list(zip(lambdas, vecs))
        eigenlist.sort(key=lambda pair: abs(pair[0]), reverse=True)

        ret = []

        for pair in eigenlist:
            lam = float(pair[0])
            vec = [float(x) for x in pair[1]]
            ret.append((lam,vec))

        return ret
    
    def printEigenset(self):
        i = 1
        for pair in self.eigenset:
            lam, vec = pair
            print(f"LAMBDA {i} = {lam}")
            print(f"EIGENVECTOR = {vec}\n")  
            i += 1

    def handlePlot(self,
                   fig,
                   ax,
                   plot_type: str = "NONE",
                   code: bool = False,
                   img: bool = False,
                   save: bool = False):

        import matplotlib.pyplot as plt

        if save:

            if   self.file == 'verysmall.json': subfolder = 'verysmall'
            elif self.file == 'small.json':     subfolder = 'small'
            elif self.file == 'large.jsonl':    subfolder = 'large'
            else:                               subfolder = 'other'

            folder = Path(__file__).parent / 'imgs' / subfolder / plot_type
            folder.mkdir(parents=True, exist_ok=True)

            filename = f"{subfolder}_{plot_type}_alpha_{self.alpha}_omega_{self.omega}.png"
            path = folder / filename

            fig.savefig(path, dpi=300, bbox_inches='tight')

        if img:
            plt.show()

        if not code:
            plt.close(fig)

        return (fig, ax) if code else None


    def getLambdaLambda(self, i: int, j: int, code=False, img=False, save=False):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,10))

        lam1, vec1 = self.eigenset[i]
        lam2, vec2 = self.eigenset[j]

        colors: dict[int, str] = {
            1: 'red',
            2: 'blue',
            3: 'green',
            4: 'orange',
            5: 'purple',
            6: 'cyan',
            7: 'magenta',
            8: 'yellow',
            9: 'brown',
            10: 'pink',
            11: 'gray',
            12: 'olive',
            13: 'navy',
            14: 'lime',
            15: 'gold',
            16: 'teal'
        }

        ax.set_title(f"x: lambda {i} = {lam1:.6f}\ny: lambda {j} = {lam2:.6f}")

        for k, (district, precinct) in enumerate(self.nodes):

            x = vec1[k]
            y = vec2[k]

            ax.scatter(x, y, color=colors.get(precinct, 'black'))

            """
            label = f"{district}{precinct}"

            plt.text(
                x,
                y,
                label,
                fontsize=9,
                ha='left',
                va='bottom'
            )
            """

        ax.axis('equal')
        ax.grid(True)

        self.handlePlot(fig,ax,'lambda_lambda',code,img,save)

        return None
    
    def getSpectralDecay(self, code=False, img=False, save=False):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,6))

        lambdas = [abs(pair[0]) for pair in self.eigenset]
        indices = [i for i in range(len(lambdas))]

        ax.set_title(f"Spectral decay for {self.file},\nalpha={self.alpha}, omega={self.omega}")
        ax.scatter(indices, lambdas)
        ax.grid(True)

        self.handlePlot(fig,ax,'spectral_decay',code,img,save)

        return None
    
    def saveInfo(self) -> None:

        if   self.file == 'verysmall.json': subfolder = 'verysmall'
        elif self.file == 'small.json':     subfolder = 'small'
        elif self.file == 'large.jsonl':    subfolder = 'large'
        else:                               subfolder = 'other'
    
        folder = Path(__file__).parent / 'results' / subfolder
        folder.mkdir(exist_ok=True)

        file_name = f"{subfolder}_alpha_{self.alpha}_omega_{self.omega}.pkl"

        path = folder / file_name

        with open(path, "wb") as f:
            pickle.dump(self.info, f)







