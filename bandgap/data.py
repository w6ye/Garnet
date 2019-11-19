
import itertools
import numpy as np
from pymacy.db import get_db
from pymatgen.core.periodic_table import get_el_sp
from .descriptor import get_average_descriptors
from .util import spe2form

garnet_site = {
    'c':['Bi3+','Hf4+','Zr4+','La3+','Pr3+','Nd3+','Sm3+',
         'Gd3+','Tb3+','Dy3+','Ho3+','Er3+','Tm3+','Yb3+','Lu3+',
         'Y3+','Cd2+','Zn2+','Ba2+','Sr2+','Ca2+','Mg2+','Na+'],
    'a':['Rh3+','Ru4+','Cr3+','Sb5+','Ta5+','Nb5+','Sn4+','Ge4+',
         'Hf4+','Zr4+','Ti4+','In3+','Ga3+','Al3+','Lu3+','Yb3+',
         'Tm3+','Er3+','Ho3+','Dy3+','Y3+','Sc3+','Zn2+','Mg2+',
         'Li+'],
    'd':['As5+','P5+','Sn4+','Ge4+','Si4+','Ti4+','Ga3+','Al3+','Li+']
}
# take out Eu3+

def get_lowest_config(entry):
    calcs = sorted(entry['calculations'], key=lambda x: entry['calculations'][x]['energy'])
    return calcs[0]


def load_unmix_data(properties,
                    sites_to_include=['c', 'a', 'd'],
                    variations=None,
                    filter={"data_type": "sub", 'youngs_modulus': {"$exists": True}}):
    """
    Args
    properties(list): properties as descriptors
    sites_to_inlucde(list): sites of the crystsal to include
    variation(list): function of variantions to put on descriptors
                eg: [np.mean, np.]
    filter(dict): qeury criteria for data in garnet_ml collection
                    default is to get all the unmix data
    """
    input_spes = []
    in_ss = []
    bandgap_ss = []
    formulas = []
    db = get_db()
    for r in db.garnet_ml.find(filter,
                               projection=["averaged_input",
                                           'a', 'c', 'd', 'elements',
                                           'band_gap(eV)']):
        if 'Yb' in r['elements']:
            continue
        if 'Eu' in r['elements']:
            continue
        if 'Rh' in r['elements']:
            continue

        if r['band_gap(eV)'] <= 0.2:
            continue

        input_spe = {"c": r['c'], 'a': r['a'], 'd': r['d']}
        inputs = [tuple([get_average_descriptors(input_spe[site], k, method='max')\
                         for k in properties])
                  for site in sites_to_include]
        if variations:
            # This returns inputs of
            # [(c_1, c_2, c_3, c_4), func1((c_1, c_2, c_3, c_4)), func2((c_1, c_2, c_3, c_4)) ...,
            #  (a_1, a_2, a_3, a_4), func1((a_1, a_2, a_3, a_4)), func2((a_1, a_2, a_3, a_4)) ...]
            inputs = [tuple(map(func, descriptors)) \
                      for descriptors in inputs \
                      for func in variations]
        # This flats the inputs, can only apply to list of TUPLEs
        inputs = list(sum(inputs, ()))
        input_spes.append(input_spe)
        in_ss.append(inputs)
        bandgap_ss.append(r["band_gap(eV)"])
    return input_spes, in_ss, bandgap_ss

def load_mix_data(mix_data,
                  properties,
                  sites_to_include=['c', 'a', 'd'],
                  variations=None):
    input_spes = []
    in_mix_gen = []
    bandgaps_mix = []
    for entry in mix_data:
        if 'analysis' not in entry:
            continue
        if 'Yb' in entry['elements']:
            continue
        if 'Eu' in entry['elements']:
            continue
        if 'Rh' in entry['elements']:
            continue

        input_spe = {"c": entry['c'], 'a': entry['a'],'d':entry['d']}
        inputs = [tuple([get_average_descriptors(input_spe[site], k, method='max') for k in properties])
                        for site in sites_to_include]
        if variations:
            # This returns inputs of
            # [(c_1, c_2, c_3, c_4), func1((c_1, c_2, c_3, c_4)), func2((c_1, c_2, c_3, c_4)) ...,
            #  (a_1, a_2, a_3, a_4), func1((a_1, a_2, a_3, a_4)), func2((a_1, a_2, a_3, a_4)) ...]
            inputs = [tuple(map(func, descriptors)) \
                      for descriptors in inputs \
                      for func in variations]
        # This flats the inputs, can only apply to list of TUPLEs
        inputs = list(sum(inputs, ()))
        in_mix_gen.append(inputs)

        lowest_config = get_lowest_config(entry)
        bg = entry['analysis'][lowest_config]['bandgap']
        bandgaps_mix.append(bg)

        input_spes.append(input_spe)
    return input_spes, in_mix_gen, bandgaps_mix


def load_create_data_set():
    c_exc = ['Hf4+', 'Cd2+']
    a_exc = ['Rh3+', 'Ru4+', 'Cr3+', 'Hf4+', 'Ti4+']
    d_exc = ['As5+', 'Ti4+']
    qualify_c_ele = [i for i in garnet_site['c'] if i not in c_exc]
    qualify_a_ele = [i for i in garnet_site['a'] if i not in a_exc]
    qualify_d_ele = [i for i in garnet_site['d'] if i not in d_exc]

    from itertools import combinations
    target = qualify_c_ele
    L = range(len(target))
    l = 2
    qualify_c_mix = [[target[x], target[y]] for x, y in combinations(L, l)] \
                    + [[target[y], target[x]] for x, y in combinations(L, l)]
    c_mix_list = [qualify_c_mix, qualify_a_ele, qualify_d_ele]

    combination = itertools.product(*c_mix_list)
    qualify_c = []
    for comb in combination:
        if comb[0] in qualify_c_mix and comb[1] in qualify_a_ele and comb[2] in qualify_d_ele:
            comb_flat = [comb[0][0], comb[0][1], comb[1], comb[2]]
            eles = [get_el_sp(spe) for spe in comb_flat]
            mole_ratio = [1, 2, 2, 3]
            charge = np.dot(np.array(mole_ratio), np.array([spe.oxi_state for spe in eles]))
            if charge == 24:
                qualify_c.append(eles)

        else:
            continue
    print(len(qualify_c))
    c_data = []
    for comb in qualify_c:
        species = {"c": {comb[0]: 1 / 3, comb[1]: 2 / 3}, "a": {comb[2]: 1}, "d": {comb[3]: 1}}
        species_str = {"c": {comb[0].__str__(): 1 / 3, comb[1].__str__(): 2 / 3}, "a": {comb[2].__str__(): 1},
                       "d": {comb[3].__str__(): 1}}
        formula = spe2form(species)
        entry = {"formula":formula,
             "type": "c_mix",
             "species": species,
             "species_str": species_str}
        c_data.append(entry)

    # Note: for A sites, C3A1A2D3 = C3A2A1D3
    from itertools import combinations
    target = qualify_a_ele
    L = range(len(target))
    l = 2
    qualify_a_mix = [[target[x], target[y]] for x, y in combinations(L, l)]

    a_mix_list = [qualify_c_ele, qualify_a_mix, qualify_d_ele]

    combination = itertools.product(*a_mix_list)
    qualify_a = []
    for comb in combination:
        if comb[0] in qualify_c_ele and comb[1] in qualify_a_mix and comb[2] in qualify_d_ele:
            comb_flat = [comb[0], comb[1][0], comb[1][1], comb[2]]
            eles = [get_el_sp(spe) for spe in comb_flat]
            mole_ratio = [3, 1, 1, 3]
            charge = np.dot(np.array(mole_ratio), np.array([spe.oxi_state for spe in eles]))
            if charge == 24:
                qualify_a.append(eles)

        else:
            continue
    print(len(qualify_a))
    a_data = []
    for comb in qualify_a:
        species = {"c": {comb[0]: 1}, "a": {comb[1]: 1 / 2, comb[2]: 1 / 2}, "d": {comb[3]: 1}}
        species_str = {"c": {comb[0].__str__(): 1}, "a": {comb[1].__str__(): 1 / 2, comb[2].__str__(): 1 / 2},
                       "d": {comb[3].__str__(): 1}}
        formula = spe2form(species)
        entry = {"formula":formula,
                 "type": "a_mix",
                 "species": species,
                 "species_str": species_str}
        a_data.append(entry)


    target = qualify_d_ele
    L = range(len(target))
    l = 2
    qualify_d_mix = [[target[x], target[y]] for x, y in combinations(L, l)] \
                    + [[target[y], target[x]] for x, y in combinations(L, l)]
    d_mix_list = [qualify_c_ele, qualify_a_ele, qualify_d_mix]

    combination = itertools.product(*d_mix_list)
    qualify_d = []
    for comb in combination:
        if comb[0] in qualify_c_ele and comb[1] in qualify_a_ele and comb[2] in qualify_d_mix:
            comb_flat = [comb[0], comb[1], comb[2][0], comb[2][1]]
            eles = [get_el_sp(spe) for spe in comb_flat]
            mole_ratio = [3, 2, 2, 1]
            charge = np.dot(np.array(mole_ratio), np.array([spe.oxi_state for spe in eles]))
            if charge == 24:
                qualify_d.append(eles)

        else:
            continue
    print(len(qualify_d))
    d_data = []
    for comb in qualify_d:
        species = {"c": {comb[0]: 1}, "a": {comb[1]: 1}, "d": {comb[2]: 2 / 3, comb[3]: 1 / 3}}
        species_str = {"c": {comb[0].__str__(): 1}, "a": {comb[1].__str__(): 1},
                       "d": {comb[2].__str__(): 2 / 3, comb[3].__str__(): 1 / 3}}
        formula = spe2form(species)
        entry = {"formula":formula,
             "type": "d_mix",
             "species": species,
             "species_str": species_str}
        d_data.append(entry)


    return c_data + a_data + d_data