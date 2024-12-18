import multiprocessing
import numpy as np
import torch
import os
from typing import Tuple, Union, List, Any
from torch.utils.data import Dataset
import re
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Manager

OMEGA = [0.0, 0.4]

RATIO = {'hartree': 1, 'kcal/mol': 627.51, 'kj/mol': 2625.5}
UNIT = 'kcal/mol'

DATASET_PARAGRAPH_SPLITTER = "****************************************\n"

ROOT = '/opt/data/private/workspace6/Data'
XC = 'DM21'
GRID = 'ta3'


class ReactionData:
    torch.set_default_dtype(torch.float64)

    def __init__(self) -> None:
        pass


class SpeciesData:
    torch.set_default_dtype(torch.float64)

    def __init__(self, partA: dict, partB: Union[np.ndarray, torch.Tensor],
                 partC: Tuple[Union[np.ndarray, torch.Tensor],
                              ...], partD: dict) -> None:
        self.summary = partA

        if isinstance(partB, np.ndarray):
            partB = torch.Tensor(partB)
        if partB.ndim != 2 or len(partB) != 4:
            raise Exception("Error in PartB")
        self.x, self.y, self.z, self.w = torch.chunk(partB, 4, dim=0)

        rho_a, rho_b, hfx_a, hfx_b, exc, etc = partC

        # features: [rhoa, rhob, g2rhoa, g2rhob, g2rhoab, taua, taub, ehfa0, ehfa1, ehfb0, ehfb1]
        #         : [lda, ehf0, ehf1, w]
        #         : [exc]
        '''============================================================'''
        rhoa = rho_a[0]
        rhob = rho_b[0]
        g2rhoa = rho_a[1]**2 + rho_a[2]**2 + rho_a[3]**2
        g2rhob = rho_b[1]**2 + rho_b[2]**2 + rho_b[3]**2
        g2rhoab = ((rho_a[1] + rho_b[1])**2 + (rho_a[2] + rho_b[2])**2 +
                   (rho_a[3] + rho_b[3])**2)
        taua = rho_a[5]
        taub = rho_b[5]
        ehfas = hfx_a
        ehfbs = hfx_b
        feature = [rhoa, rhob, g2rhoa, g2rhob, g2rhoab, taua, taub]
        feature.extend(ehfas)
        feature.extend(ehfbs)
        self.feature = torch.tensor(np.array(feature)).type(torch.float64)
        '''============================================================'''
        lda = np.array(-2 * np.pi * (3 / 4 / np.pi * (rhoa + rhob))**(4 / 3))
        ehfa, ehfb = list(zip(ehfas, ehfbs))
        ehfa = np.sum(np.array(ehfa), axis=0)
        ehfb = np.sum(np.array(ehfb), axis=0)
        self.energy = torch.tensor(np.array([lda, ehfa,
                                             ehfb])).type(torch.float64)
        '''============================================================'''
        self.vxc = torch.tensor(exc).type(torch.float64)
        self.exc = self.vxc * (rhoa+rhob)
        self.etc = torch.tensor(etc).type(torch.float64)  # [del]unweighted exc potential or empty[/del]

        # if self.etc.shape == (1, 0):
        #     self.vxc = None
        # else:
        #     if self.etc.ndim == 1:
        #         self.vxc = self.etc
        #     elif self.etc.ndim == 2:
        #         self.vxc = self.etc[0]
        #     else:
        #         raise Exception

        self.atomset = partD

        self.len = len(self.exc)

    def getweight(self):
        return self.w

    def getfullenergy(self):
        return self.summary['e_tot']

    def getexc(self):
        return self.exc

    def getinputdata(self, version: int = 7):
        '''version is a dec number converted from a bin number'''
        '''descending: with energies, with weight, with exc'''
        flags = [bool(int(item)) for item in str(bin(version))[2:]]
        datalist = [self.feature]

        if flags[0]:  # energies
            datalist.append(self.energy)

        if flags[1]:  # weight
            w = self.w
            if w.ndim != 2:
                w = torch.reshape(w, (1, -1))
            datalist.append(w)

        if flags[2]:  # exc
            vxc = self.vxc
            if vxc.ndim != 2:
                vxc = torch.reshape(vxc, (1, -1))
            datalist.append(vxc)

        return torch.concat(datalist)

    def getlabel(self):
        # True should throw, and False should keep
        # weighted exc and vxc are the two gates

        exc = torch.abs(self.exc * self.w).squeeze()  # weighted exc
        # single exc
        # we can remove no more than REDUCEGATE percentage of list

        REDUCEGATE = 0.5
        # REDUCEGATE = 1  # temporary change in this try
        SUMGATER = 1E-3  # no more than Ha error per Ha, or no more than 0.1% change
        SUMGATEN = 1 / 627.51  # at least 1 kcal / mol error in total is allowed

        gate_index1 = int(len(exc) * REDUCEGATE)

        # cumsum exc
        # sum of abs of removed points should not bigger than SUMGATER percentage or SUMGATEN number
        # that means, the sum could be 1 kcal/mol at least
        exc_asc = torch.sort(exc, dim=-1, descending=False).values
        exc_cum = torch.cumsum(exc_asc, dim=-1)  # exc_sum is asc
        cum_max = np.max((float(exc_cum[-1]) * SUMGATER, SUMGATEN))
        gate_index2 = int(
            torch.argmin(torch.abs(exc_cum - cum_max), dim=0)
        )  # remove gate_indexth points, in other words, remove no more than gate_index number of points
        gate_index = min(gate_index1, gate_index2)
        gate = torch.kthvalue(exc, gate_index)
        exc_label = torch.where(exc < gate[0], True, False)
        # counts =[ torch.count_nonzero(exc_label), torch.count_nonzero(~exc_label)]
        # print(list(zip(exc_label,exc)))

        if self.vxc != None:
            vxc = torch.abs(self.vxc * self.w)  # weighted vxc
            vxc = torch.squeeze(vxc)
            # single vxc
            # we can remove no more than REDUCEGATER percentage of list and no bigger than REDUCEGATEN

            REDUCEGATER = 0.3
            REDUCEGATEN = 1E-2
            gate_max1 = torch.kthvalue(vxc, int(len(vxc) * REDUCEGATER))
            gate_max = min(gate_max1[0], REDUCEGATEN)
            vxc_label = torch.where(vxc < gate_max, True, False)
            # s = [1 for item in vxc_label if item]
            # count = exc_label & vxc_label
            # s = [1 for item in count if item]
            # # [print(item[0]) for item in list(zip(exc,count)) if item[1]]
            # [print(item[0]) for item in list(zip(vxc,count)) if item[1]]
        else:
            vxc_label = torch.ones_like(exc_label).type(torch.bool)
        return torch.stack((exc_label, vxc_label), dim=0).T


class DatasetNewNet(Dataset):
    def __init__(self,
                 datafile: str,
                 size: int = 1048576,
                 version: int = 7) -> None:
        super().__init__()
        datalist = read_specs(filepath=datafile)
        if size < len(datalist):
            datalist = datalist[:size]

        # plan is a dict of parsed reaction plan
        self.plan = parse_plan(datalist)
        # data is list of loaded data dict
        self.data = generate_data(self.plan)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index) -> Any:
        return self.plan[index]


def generate_data(plan: List[dict]):
    specieslist = []
    output = {}
    for reaction in plan:
        for species in reaction:
            if species == 'gt':
                continue
            specieslist.append(species)
            specieslist.append(re.sub('_[0-9]*@', '_f@', species))
    specieslist = list(set(specieslist))
    specieslist.sort()

    def read(species):
        mol, datapool = str(species).split('@')
        datapool = datapool.upper()

        filepath = os.path.join(ROOT, XC, datapool, GRID,
                                f"{mol}.{(XC+GRID).lower()}")
        rawdata = prase_file(filepath)
        partA, partB, partC, partD = rawdata
        try:
            data = SpeciesData(partA, partB, partC, partD)
            print(f"{species} has been loaded", end='\n')
        except:
            raise Exception(f"{species}")

        return {species: data}

    pool = Pool(80)
    result = pool.map(read, specieslist)
    [output.update(item) for item in result]

    return output


def prase_file(filepath: str):
    torch.set_default_dtype(torch.float64)
    with open(filepath) as f:
        datalines = f.read()
    if datalines == []:
        raise Exception(f"{filepath} is empty")
    '''
    file structure:
    L01 DATASET_PARAGRAPH_SPLITTER
    L02 name
    L03 e_tot
    L04 e1
    L05 coul
    L06 exc
    L07 nuc
    L08 DATASET_PARAGRAPH_SPLITTER
    L09 x
    L10 y
    L11 z
    L12 w
    L13 DATASET_PARAGRAPH_SPLITTER
    L14 rho_a
    L15 rho_a
    L16 rho_a
    L17 rho_a
    L18 rho_a
    L19 rho_a
    L20 rho_b
    L21 rho_b
    L22 rho_b
    L23 rho_b
    L24 rho_b
    L25 rho_b
    L26 hfx_a
    L27 hfx_a
    L28 hfx_b
    L29 hfx_b
    L30 exc
    L?? etc
    L-3 DATASET_PARAGRAPH_SPLITTER
    L-2 atomlist
    L-1 DATASET_PARAGRAPH_SPLITTER
    '''
    partA, partB, partC, partD, = [
        item for item in datalines.split(DATASET_PARAGRAPH_SPLITTER)
        if item != ''
    ]

    partA = [item for item in partA.split('\n') if item != '']
    summary = {}
    name, e_tot, e1, coul, exc, nuc = partA
    summary['name'] = name
    summary['e_tot'] = float(e_tot)
    summary['e1'] = float(e1)
    summary['coul'] = float(coul)
    summary['exc'] = float(exc)
    summary['nuc'] = float(nuc)

    partB = [item for item in partB.split('\n') if item != '']
    x, y, z, w = partB
    x = list(map(eval, [item for item in x.split(',') if item != '']))
    y = list(map(eval, [item for item in y.split(',') if item != '']))
    z = list(map(eval, [item for item in z.split(',') if item != '']))
    w = list(map(eval, [item for item in w.split(',') if item != '']))
    coords = np.array([x, y, z, w])

    partC = [item for item in partC.split('\n') if item != '']
    rho_a = partC[0:6]
    rho_b = partC[6:12]
    hfx_a = partC[12:14]
    hfx_b = partC[14:16]
    rho_a = np.array([[float(cell) for cell in item.split(',') if cell != '']
                      for item in rho_a])
    rho_b = np.array([[float(cell) for cell in item.split(',') if cell != '']
                      for item in rho_b])
    hfx_a = np.array([[float(cell) for cell in item.split(',') if cell != '']
                      for item in hfx_a])
    hfx_b = np.array([[float(cell) for cell in item.split(',') if cell != '']
                      for item in hfx_b])

    exc = []
    etc = []
    if len(partC) == 17:  # has exc no etc
        exc = partC[16]
        exc = list(map(eval, [item for item in exc.split(',') if item != '']))

    elif len(partC) >= 18:  # has exc etc is detail
        exc = partC[16]
        exc = np.array(
            list(map(eval, [item for item in exc.split(',') if item != ''])))

        for line in partC[17:]:
            etc.append(
                np.array(list(
                    map(eval,
                        [item for item in line.split(',') if item != '']))))

    else:
        raise Exception
    exc = np.array(exc)
    etc = np.array(etc)

    atomline = partD[2:-3]
    atomline = atomline.split('\', \'')
    atomlist = [item.split(',') for item in atomline]
    atomset = {item[0]: item[1] for item in atomlist}

    return summary, coords, (rho_a, rho_b, hfx_a, hfx_b, exc, etc), atomset


def parse_plan(datalist: list) -> List[dict]:
    output = []
    for dataline in datalist:
        cell = {}
        # sturcture of dataline: type, species, coeff, energy, dataset
        type, species, coeff, energy, dataset = dataline

        # type 0 means energy is meaningless
        # type 1 means Ha, type 2 means kcal/mol, type 3 means kj/mol
        # type 4 reserves for eV

        if type == 0:
            energy = 0.
        elif type == 1:
            energy *= RATIO[UNIT.lower()] / RATIO['hartree']
        elif type == 2:
            energy *= RATIO[UNIT.lower()] / RATIO['kcal/mol']
        elif type == 3:
            energy *= RATIO[UNIT.lower()] / RATIO['kj/mol']
        else:
            raise Exception
        cell.update({'gt': energy})

        for item, mul in list(zip(species, coeff)):
            cell.update({f"{item}@{dataset.lower()}": mul})
        output.append(cell)

    if len(output) != 0:
        return output
    else:
        raise Exception


def read_specs(filepath: str) -> list:
    """
    read specification file, ext = '.plan'

    exampleA:
    A + B == 3C Energy = n kJ/mol, in XXX dataset
    in file
    $t\t1\t$m\tA\tB\tC\t$w\t-1\t-1\t3\t$e\tn\t$s\tXXX\n
    output
    [1,[A,B,C],[-1,-1,3],n,XXX]

    exampleB:
    A Energy = n kJ/mol, in XXX dataset
    $t\t1\t$m\tA\t$w\t1\t$e\tn\t$s\tXXX\n
    output
    [1,[A],[1],n,XXX]
    """

    torch.set_default_dtype(torch.float64)
    output = []
    with open(filepath) as f:
        datalines = f.read().split('\n')
    datalines = [item.split('\t') for item in datalines if item != '']

    for i in range(len(datalines)):
        dataline = datalines[i]

        # ignore line starting with #
        if dataline[0][0] == '#':
            continue

        if '@' in dataline:
            raise Exception("@ is reserved char")

        dataline = [item for item in dataline if item != '']

        bookmark = {'$m': -1, '$t': -1, '$w': -1, '$e': -1, '$s': -1}
        for j in range(len(dataline)):
            if dataline[j] == '$m':
                bookmark['$m'] = j
                continue
            if dataline[j] == '$t':
                bookmark['$t'] = j
                continue
            if dataline[j] == '$w':
                bookmark['$w'] = j
                continue
            if dataline[j] == '$e':
                bookmark['$e'] = j
                continue
            if dataline[j] == '$s':
                bookmark['$s'] = j
                continue

        bookmark = list(bookmark.values())
        bookmark.append(len(dataline))
        bookmark.sort()

        if -1 in bookmark:
            print(f"line {i} is not intact")
            continue

        pieces = []
        for j in range(len(bookmark) - 1):
            pieces.append(dataline[bookmark[j]:bookmark[j + 1]])

        species, type, coeff, energy, dataset = [], [], [], [], []
        for item in pieces:
            if item[0] == '$m':
                species = item[1:]
            elif item[0] == '$t':
                type = item[1:]
            elif item[0] == '$w':
                coeff = item[1:]
            elif item[0] == '$e':
                energy = item[1:]
            elif item[0] == '$s':
                dataset = item[1:]
        try:
            if len(type) != 1 or len(energy) != 1 or len(dataset) != 1:
                raise Exception
            type = int(type[0])
            coeff = [int(item) for item in coeff]
            # type 0 means energy is meaningless
            # type 1 means Ha, type 2 means kcal/mol
            # type 3 reserves for eV

            energy = float(energy[0])
            dataset = dataset[0]
        except:
            print(f"line {i} is not intact")
            continue
        output.append([type, species, coeff, energy, dataset])

    return output


if __name__ == "__main__":
    a = prase_file("/opt/data/private/workspace6/Data/DM21/W4-17/ta3/al_f.dm21ta3")
    b = SpeciesData(*a)
    pass
