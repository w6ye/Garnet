# coding: utf-8

import numpy as np
import pandas as pd
import argparse
import re
import os
import json
import pickle

from keras.models import load_model

from pymacy.qe import MVLQE
from pymatgen import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram


from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.sets import _load_yaml_config

"""
This python codes is designed for predicting the formation
energy referenced to stable binary oxides (Ef) and energy abo-
ve convex hull(Ehull) of unknown garnet materials. C, A and D
represents 24c, 16a and 24d Wyckoff sites in prototypical cu-
bic garnet structure Y3Al5O12(ICSD 170157, spacegroup Ia-3d).
The oxidation state is pre-assigned to be common oxidation state
of the elements (see Table 1 in paper). The codes takes input
of formula on each site as well as the path of the model and
corresponding scaler files.

How to use :
1. Prepare corresponding model.h5 and StandardScaler.pkl files
provided with the paper.
2. The inputs takes the formula on each site as well as the path
to the model.h5 and StandardScaler.pkl file.

For example:

predict_garnet_stability.py -C Y3 -A Al2 -D Al3 -m "model.h5" -s "scaler.pkl"

returns Ef (eV/f.u.),Ehull (eV/atom) of Y3Al5O12 garnet material

"""

__author__ = "Weike Ye"
__version__ = "1.0"
__maintainer__ = "Weike Ye"
__email__ = "w6ye@ucsd.edu"
__date__ = "Oct 16 2017"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

els_path = os.path.join(MODULE_DIR, 'tools/elements.json')
with open(els_path, 'r') as f:
    garnet_elements = json.load(f)

# ox_path = os.path.join(MODULE_DIR, 'tools/ox_table.json')
# with open(ox_path, 'r') as f:
#     ox_table = json.load(f)


def spe2form(species):
    """
    Transfer from species dict to standard garnet formula
    :param speciess: dict
            species in dictionary
            e.g. for Y3Al5O12,
            species={"c":{"Y":3},
                    "a":{"Al":2},
                    "d":{"Al":3}},
            for BiBa2Hf2Ga3O12,
            species={"c":{"Bi":1,"Ba":2},
                    "a":{"Hf":2},
                    "d":{"Ga":3}}
    :return:
    """
    sites = ['c', 'a', 'd']
    spe_list = [re.sub(r"[^A-Za-z]+", '', el) + str(amt)
                for site in sites for el, amt in species[site].items()]
    formula = "".join(spe_list)
    formula.replace("1", "")
    num_cations = sum([amt for site in sites
                       for el,amt in species[site].items()])
    num_O = 12 * num_cations/8
    return formula + "O%s"%num_O


def model_load(model_path, scaler_path):
    """
    Load model and scaler for Ef prediction
    :param model_path: str
            path to model.h5 file
    :param scaler_path: str
            path to scaler.pkl file
    :return: (model, scaler)
    """
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def get_X_one(spe):
    """
    Get electronegativity for one species/ion
    :param el: str
            specie in string, e.g. 'Al3+' or 'Al'
    :return: float
            electronegtivity
    """
    # make sure spe does not contain charge
    regex = re.compile('[^a-zA-Z]')
    el = regex.sub('', spe)
    #
    # df_eleneg = pd.DataFrame.from_csv(os.path.join(MODULE_DIR,
    #                                   'tools/electron_ng.csv'))
    # eleneg = df_eleneg.to_dict(orient='record')
    # for item in eleneg:
    #     if item['Element'] == spe:
    #         return item['Electronegtivity(Pauling Scale)']
    return get_el_sp(el).X

def get_X_mixed(spe):
    """

    :param
        spe: str or dict
            specie in string or dict in the format
            {site:{el1:amt1,el2:amt2}}
    :return: float
            electronengtivity if spe is dict,
            return the mean of electronegtivity
            from definition (Binding energy)
            cf https://www.wikiwand.com/en/Electronegativity
    """
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


def get_ir_one(ion):
    """
    Get ionic radii for one species/ion
    :param ion: eg Al3+
    :return: float
            inoic radius of el
    """
    if any(char.isdigit() for char in ion) \
            or '+' in ion \
            or '-' in ion:
        ele = get_el_sp(ion)
    else:
        ele = get_el_sp([i for i in garnet_elements
                         if ion == re.split(r'(\d+)', i)[0]][0])
    return ele.ionic_radius  # , ele.oxi_state

def get_ir_mixed(species):
    """

    :param species: str or dict
        element in str or a dictionary,
        the oxidation state is defined
        in garnet elements unless otherwise specified,

    :return: float
        ionic radius, if species is in dict,
        return weighted mean ir
    """

    if type(species) == str:
        return get_ir_one(species)
    if type(species) == dict:
        ir_avg = 0
        factor = sum([species[el] for el in species])
        for el in species:
            ir_avg += get_ir_one(el) * species[el] / factor
        return ir_avg


def get_input(species):
    """
    Prepare the inputs for model prediction
    i.e. extract the [XA, rA, XC, rC, XD, rD] inputs array
    for given species
    :param species: dict
    :return: inputs array: dict
            {"X_a":float,"r_a":float,
            "X_c":float,"r_c":float,
            "X_d":float,"r_d":float}
    """
    inputs = {}
    for site in ['a', 'd', 'c']:
        spes = species[site]
        x = get_X_mixed(spes)
        ir = get_ir_mixed(spes)
        inputs.update({'X_%s' % site: x, "r_%s" % site: ir})
    return inputs


def get_form_e(inputs, model, scaler):
    """
    Get formation energy from given inputs
    :param inputs: list
           input list in the form of
           ["X_a", "r_a", "X_c", "r_c", "X_d", "r_d"]
    :param model: keras model object
    :param scaler: keras StandardScaler object
    :return: predicted Ef, float
    """
    keys = ["X_a", "r_a", "X_c", "r_c", "X_d", "r_d"]
    inputs_n = [inputs[key] for key in keys]
    inputs_scaled = scaler.transform(inputs_n)
    return model.predict(inputs_scaled.reshape(1, 6))[0][0]


def get_tote(form_e, species):
    """
    get total energy with respect to given Ef and species
    :param form_e: float
           should be in accordance with speciees.
           e.g. the form_e is in eV/f.u.,
                the species should be given in a formula unit.
    :param species: dict
            e.g. {"c":{"Zr4+":1,"Zn2+":2},
                  "a":{"Mg2+":2},
                  "d":{"Si4+":3}}
    :return: tot_e, float
            in the same unit as form_e
    """
    m = MPRester()
    formula = spe2form(species)
    composition = Composition(formula)
    tote = form_e
    for el, amt in composition.items():
        stable_bio_entry = None
        if el.symbol == 'O':
            continue
        if el.symbol == 'Yb':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR,
                             'tools/mp-2814_Yb16O24_vasprun.xml.relax2.gz')).get_computed_entry()
        if el.symbol == 'Nb':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR,
                             'tools/ICSD_25750_Nb2O5_vasprun.xml.relax2.gz')).get_computed_entry()
        if el.symbol == 'Eu':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR,
                             'tools/ICSD_40472_Eu2O3_vasprun.xml.relax2.gz')).get_computed_entry()

        if not stable_bio_entry:
            stable_bio_df = pd.DataFrame.from_csv(os.path.join(MODULE_DIR,
                                                               'tools/stable_binary_oxides_garnet.csv'))
            stable_bio_id = stable_bio_df.loc[lambda df: df.specie == el.symbol]['mpid'].tolist()[0]
            stable_bio_entry = m.get_entry_by_material_id(stable_bio_id,
                                                          property_data=['e_above_hull',
                                                                         'formation_energy_per_atom'])
        min_e = stable_bio_entry.uncorrected_energy
        amt_bio = stable_bio_entry.composition[el.name]
        tote += (amt / (amt_bio)) * min_e

    return tote


def get_ehull(tot_e, species=None, formula=None,
              db_path=os.path.join(MODULE_DIR, 'tools/cal_db.json')):
    """
    get ehull predicted under given total energy and species.
    The composition can be either given by the species dict(for garnet
    only) or a formula
    :param tot_e: float
            the unit of total energy in accordance with
            given composition
    :param species: dict
    :param formula: str
            formula of given compound, e.g. "Ca3Sc2Ge4O12"
    :param db_path: str
            database config file
    :return: Ehull: float
             in eV/atom
    """
    if not any([species, formula]):
        raise ValueError("Invalid Input: must provide species or formula")

    CONFIG = _load_yaml_config("MPRelaxSet")
    LDAUU = CONFIG["INCAR"]['LDAUU']['O']

    if species:
        #    sites = ['c', 'a', 'd']
        formula = spe2form(species)
    # elements = [re.sub(r"[^A-Za-z]+", '', el)
    #                for site in sites for el, amt in species[site].items()] + ['O']

    composition = Composition(formula)
    elements = [el.name for el in composition]

    mv = MVLQE(db_path=db_path)
    compat = MaterialsProjectCompatibility()
    all_entries = mv.get_entries_in_system(elements=elements, optional_data=['task_id'])

    all_entries = compat.process_entries(all_entries)

    potcars = set()
    for e in all_entries:
        if len(e.composition) == 1 and e.composition.reduced_formula in elements:
            potcars.update(e.parameters["potcar_symbols"])

    potcars.update({"pbe O"})
    parameters = {"potcar_symbols": list(potcars)}

    for el in elements:
        if el in LDAUU:
            parameters.update({"hubbards": {el: LDAUU[el]}})

    ec = ComputedEntry(composition=composition, energy=0, parameters=parameters)
    ec.uncorrected_energy = tot_e
    ec = compat.process_entry(ec)

    pd = PhaseDiagram(all_entries + [ec])
    #analyzer = PDAnalyzer(pd)

    #decomp, ehull = analyzer.get_decomp_and_e_above_hull(ec)

    return pd.get_e_above_hull(ec)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
         This python codes is designed for predicting Ef and Energy above
         convex hull(Ehull) of unknown garnet materials. C, A and D
         represents 24c, 16a and 24d Wyckoff site in prototypical cu-
         bic garnet structure Y3Al5O12(ICSD 170157, spacegroup Ia-3d).
         The inputs takes the formula on each site as well as the path
         to the model.h5 and StandardScaler.pkl file.
         An example:
         predict_garnet_stability.py -C Y3 -A Al2 -D Al3 -m "model.h5" -s "scaler.pkl"
         returns the Ef and Ehull of Y3Al5O12 garnet material.""")
    parser.add_argument('-C', '--c',
                        help="elements on C site followed by the amt,"
                             "e.g. 'Ca3','Ca2Y'",
                        required=True)
    parser.add_argument('-A', '--a',
                        help="elements on A site followed by the amt,"
                             "e.g. 'Sc2','ScMg'",
                        required=True)
    parser.add_argument('-D', '--d',
                        help="elements on D site followed by the amt,"
                             "e.g. 'Ge3','Ge2Si'",
                        required=True)
    parser.add_argument('-m', '--model',
                        help="The directory of model.h5 file",
                        required=True)
    parser.add_argument('-s', '--scaler',
                        help="The directory of scaler.pkl file",
                        required=True)
    # parser.add_argument('-db', '--dbconfig', help="Config file to use. If"
    #                         "none is found, an no-authentication "
    #                         "localhost:27017/vasp database and tasks "
    #                         "collection is assumed.")
    args = parser.parse_args()
    species = {}
    for arg, inputs in args._get_kwargs():
        if arg not in ['a', 'c', 'd']:
            continue
        str_parse = re.findall(r'([A-Z][a-z]*)(\d*)', inputs)
        species[arg] = {el: int(amt) if amt else 1 for el, amt in str_parse}

    model, scaler = model_load(args.model, args.scaler)
    inputs = get_input(species)
    form_e = get_form_e(inputs, model, scaler)
    tote = get_tote(form_e, species)
    ehull_pred = get_ehull(tote, species=species)
    print(inputs)
    print(form_e, tote, ehull_pred)
