from pymacy.icsd.db import ICSD
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy

import pandas as pd
import numpy as np
import os
import json
import re


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
garnet_site = {
    'c': ['Bi3+', 'Hf4+', 'Zr4+', 'La3+', 'Pr3+', 'Nd3+', 'Sm3+', 'Eu3+',
          'Gd3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+', 'Lu3+',
          'Y3+', 'Cd2+', 'Zn2+', 'Ba2+', 'Sr2+', 'Ca2+', 'Mg2+', 'Na1+'],
    'a': ['Rh3+', 'Ru4+', 'Cr3+', 'Sb5+', 'Ta5+', 'Nb5+', 'Sn4+', 'Ge4+',
          'Hf4+', 'Zr4+', 'Ti4+', 'In3+', 'Ga3+', 'Al3+', 'Lu3+', 'Yb3+',
          'Tm3+', 'Er3+', 'Ho3+', 'Dy3+', 'Y3+', 'Sc3+', 'Zn2+', 'Mg2+',
          'Li1+'],
    'd': ['As5+', 'P5+', 'Sn4+', 'Ge4+', 'Si4+', 'Ti4+', 'Ga3+', 'Al3+', 'Li1+']
}
with open(os.path.join(MODULE_DIR, 'tools/elements.json'), 'r') as f:
    garnet_elements = json.load(f)

json_path = os.path.join(MODULE_DIR, 'tools/ox_table.json')

with open(json_path, 'r') as f:
    ox_table = json.load(f)


def is_neutral(entry, cation):
    """

    :param entry: pymatgen.ComputedEntry
    :param cation: str
            e.g. ['Ag']
    :return: bool
            True if neutral otherwise False
    """

    amt_o = entry.composition['O']
    amt_c = entry.composition[cation]
    charge = int(ox_table[cation]) * amt_c + (-2) * amt_o
    return True if not charge else False


def get_mpid(icsd_id):
    """

    :param icsd_id: int
    :return: str
            'not in mp' if there is no corresponding entry in MP
            else mpid ('mp-XXXX')
    """
    db = ICSD(collection_name='icsd_2017_v1_unique')
    query = db.query({'icsd_ids': {'$in': [icsd_id]}}, ['icsd_ids', 'mp'])
    result = [i for i in query][0]
    if 'mp' not in result:
        return 'not in mp'
    else:
        return (result['mp']['id'])

def get_cn_sites(structure, exclude_ele=['O'], maximum_distance_factor=1.5):
    """

    :param structure: Pymatgen structure Object
    :param exclude_ele: list of elements not to be considered, eg ['O']
    :param maximum_distance_factor:
    :return: a dictionary in the format {cn_1:[sites with coordination number of cn_1]}
    """
    lgf = LocalGeometryFinder()
    lgf.setup_parameters(structure_refinement='none')
    lgf.setup_structure(structure)
    se = lgf.compute_structure_environments(maximum_distance_factor=maximum_distance_factor)
    default_strategy = SimplestChemenvStrategy(se)
    cn_sites = {}
    for eqslist in se.equivalent_sites:
        eqslist = [i for i in eqslist if i.specie.symbol not in exclude_ele]
        if not eqslist:
            continue
        site = eqslist[0]
        ces = default_strategy.get_site_coordination_environments(site)
        ce = ces[0]
        cn = int(ce[0].split(':')[1])
        if cn in cn_sites:
            cn_sites[cn].extend(eqslist)

        else:
            cn_sites.update({cn: [site for site in eqslist]})

    return cn_sites

def get_site_spe_from_structure(s, maximum_distance_factor=1.5):
    """

    :param s: Pymatgen.Structure
    :param maximum_distance_factor:
    :return: dict
        e.g. {"c":{"Y":2,"Lu":1},"d":{"Al":3},"a":{"Al":2}}


    """

    cn_sites = get_cn_sites(s)
    site_spes = {}
    cn2site = {8:"c",4:"d",6:"a"}
    for cn in cn_sites:
        spes = [i.specie.name for i in cn_sites[cn]]
        spes_occ = {spe:spes.count(spe)/4 for spe in spes}
        site_spes.update({cn2site[cn]:spes_occ})

    return site_spes


def get_site_spe_from_formula(formula):
    """

    :param formula: str
            The formula of the structure in the format of C3A2D3O12
    :return: return dict
        e.g. {"c":{"Y":2,"Lu":1},"d":{"Al":3},"a":{"Al":2}}
    """

    m = re.findall(r"([A-Z][a-z]*)\s*([-*\.\d]*)", formula)
    m = [(el, int(r) if r != "" else 1) for el, r in m]
    lst = []
    for el, r in m:
        lst += [el] * r
    c_lst = lst[:3]
    a_lst = lst[3:5]
    d_lst = lst[5:8]
    a = {}
    c = {}
    d = {}
    for el in set(a_lst):
        a.update({el: a_lst.count(el)})
    for el in set(c_lst):
        c.update({el: c_lst.count(el)})
    for el in set(d_lst):
        d.update({el: d_lst.count(el)})
    return {'a': a, 'c': c, 'd': d}


def get_ir_one(el):
    """

    :param el: eg Al3+
    :return: float
            inoic radius of el
    """
    if any(char.isdigit() for char in el) or '+' in el or '-' in el:
        ele = get_el_sp(el)
    else:
        ele = get_el_sp([i for i in garnet_elements if el == re.split(r'(\d+)', i)[0]][0])
    return ele.ionic_radius  # , ele.oxi_state


def get_ir_mixed(spes):
    """

    :param spe: str or dict
        element in str or a dictionary, the oxistate is defined in garnet elements unless otherwise specified,

    :return: float
        ionic radius, if spe is in dict, return weighted mean ir
    """

    if type(spes) == str:
        return get_ir_one(spes)

    if type(spes) == dict:
        mean_ir = 0

        factor = sum([spes[el] for el in spes])

        for el in spes:
            mean_ir += get_ir_one(el) * spes[el] / factor
        return mean_ir


def get_X_one(el):
    """

    :param spe: str
            specie in string, e.g. 'Al3+'
    :return: float
            electronegtivity
    """
    # make sure spe does not contain charge
    regex = re.compile('[^a-zA-Z]')
    spe = regex.sub('', el)

    df_eleneg = pd.DataFrame.from_csv(os.path.join(MODULE_DIR, 'tools/electron_ng.csv'))
    eleneg = df_eleneg.to_dict(orient='record')
    for item in eleneg:
        if item['Element'] == spe:
            return item['Electronegtivity(Pauling Scale)']


def get_X_weighted_mixed(spes):
    """

    :param spe: str or dict
        specie in string or dict
    :return: float
        electronegtivity. if spe is dict, return weighted mean electronegtivity
    """
    # make sure spe does not contain charge
    if type(spes) == str:
        return get_X_one(spes)

    if type(spes) == dict:
        mean_en = 0

        factor = sum([spes[el] for el in spes])
        for el in spes:
            mean_en += get_X_one(el) * spes[el] / factor
        return mean_en


def get_X_BD_mixed(spe):
    """

    :param
        spe: str or dict
            specie in string or dict in the format {site:{el1:amt1,el2:amt2}}
    :return: float
            electronengtivity if spe is dict, return the mean of electronegtivity from definition (Binding energy)
            cf https://www.wikiwand.com/en/Electronegativity
    """
    # make sure spe does not contain charge
    o = get_el_sp('O2-')

    num_sites = sum([v for k, v in spe.items()])
    if len(spe) < 2:
        el = get_el_sp(list(spe.keys())[0])
        return el.X
    else:
        avg_eneg = 0

        for s, amt in spe.items():
            el = get_el_sp(s)
            avg_eneg += (amt / num_sites) * (el.X - o.X) ** 2

        return np.abs(o.X - (np.sqrt(avg_eneg)))


def get_averaged_input(site_spes, x_average='BD'):
    """

    :param site_spes: dict
            e.g. {"c":{"Y":2,"Lu":1},"d":{"Al":3},"a":{"Al":2}}
    :param x_average: str "BD" or "weighted"
            how the mean electronegtivity is calculated
    :return: dict
            averaged input for model prediction
            {"c_eneg":,"c_radius":,"a_eneg":,"a_radius":,"d_eneg":,"d_radius":}
    """
    averaged_input = {}
    for site in ['a', 'd', 'c']:
        spes = site_spes[site]
        if x_average == 'BD':
            x = get_X_BD_mixed(spes)
        elif x_average == 'weighted':
            x = get_X_weighted_mixed(spes)
        ir = get_ir_mixed(spes)
        averaged_input.update({'%s_eneg' % site: x, "%s_radius" % site: ir})
    return averaged_input
