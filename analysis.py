from pymatgen.io.vasp import Vasprun
from pymatgen.phasediagram.maker import PhaseDiagram
from pymatgen.phasediagram.analyzer import PDAnalyzer
from pymacy.qe import MVLQE
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
import json

import pandas as pd
import os
import re

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(MODULE_DIR, 'tools/ox_table.json')
with open(json_path, 'r') as f:
    ox_table = json.load(f)


def get_pred_ehull(tote_calc, form_e_predict, form_e_calc, elements, composition,
                        dbpath=os.path.join(MODULE_DIR, 'tools/cal_db.json')):
    """
    calculate ehull from predicted form_e_from_binary
    form_e_from_binary_calc = tot_e_calc - sum(amt * binary_oxide_tote)
    form_e_from_binary_pred = tot_e_pred - sum(amt * binary_oxide_tote)
    tot_e_pred = tot_e_calc + (form_e_from_binary_pred - form_e_from_binary_calc)
    from tot_e_pred, created entry and calculate synthetic_ehull

    :param tote_calc: float
            DFT calculated total e, eV/supercell
    :param form_e_predict: float
            predicted form_e_from_binary from model
    :param form_e_calc: float
            calculated form_e_from_binary from tote_calc
    :param elements: list
            list of elements in formula
    :param composition: str
            formula
    :param dbpath: db path config file
    :return: tuple (tote_pred,ehull_pred)
            tote_pred in eV/supercell, pred_ehull in eV/atom
    """
    # create entry
    compat = MaterialsProjectCompatibility()
    mv = MVLQE(db_path=dbpath)
    all_entries = mv.get_entries_in_system(elements=elements, optional_data=['task_id'])
    all_entries = compat.process_entries(all_entries)
    potcars = set()
    for e in all_entries:
        if len(e.composition) == 1 and e.composition.reduced_formula in elements:
            potcars.update(e.parameters["potcar_symbols"])

    ec = ComputedEntry(composition=composition, energy=0, parameters={"potcar_symbols": list(potcars)})
    ec.uncorrected_energy = tote_calc + (form_e_predict - form_e_calc)
    ec = compat.process_entry(ec)

    pd = PhaseDiagram(all_entries + [ec])
    analyzer = PDAnalyzer(pd)

    decomp, ehull = analyzer.get_decomp_and_e_above_hull(ec)

    return ec.uncorrected_energy, ehull


def get_ehull(entry_dict, dbpath=os.path.join(MODULE_DIR, 'tools/cal_db.json'), corrected=False):
    """

    :param entry_dict:  dic
        vasprun.get_computed_entry().as_dict()
    :param dbpath: str
        the db_config file dir
    :param corrected: bool
        False for not doing correction on the total energy
    :return: float
        Ehull(eV/atom)
    """

    entry = ComputedEntry.from_dict(entry_dict)

    elements = [e.symbol for e in entry.composition.keys()]
    mv = MVLQE(db_path=dbpath)
    entries = mv.get_entries_in_system(elements=elements, optional_data=['task_id'])
    exclude_list = None

    compat = MaterialsProjectCompatibility()
    entries = compat.process_entries(entries)
    if not corrected:
        entry = compat.process_entry(entry)

    if exclude_list is not None:
        entries = filter(lambda e: e.composition.reduced_formula not in exclude_list, entries)

    pd = PhaseDiagram(entries + [entry])
    analyzer = PDAnalyzer(pd)

    decomp, ehull = analyzer.get_decomp_and_e_above_hull(entry)

    return (ehull)


def get_form_e_from_bio(entry_dict, factor='per_fu', energy=None):
    """

    :param entry_dict:dict
        vasprun.get_computed_entry().as_dict()
    :param factor: string 'per_fu' or 'per_atom'
        the unit of the formation energy
    :param energy: float
        total energy of the entry, if not given ,use entry.uncorrected_energy
    :return: float
        formation energy from binary oxides (eV/atom)
    """
    entry = ComputedEntry.from_dict(entry_dict)
    if energy:
        bio_form_e = energy
    else:
        bio_form_e = entry.uncorrected_energy

    m = MPRester()

    for el, amt in entry.composition.items():
        stable_bio_entry = None
        if el.symbol == 'O':
            continue
        if el.symbol == 'Yb':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/mp-2814_Yb16O24_vasprun.xml.relax2.gz')).get_computed_entry()
        if el.symbol == 'Nb':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/ICSD_25750_Nb2O5_vasprun.xml.relax2.gz')).get_computed_entry()
        if el.symbol == 'Eu':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/ICSD_40472_Eu2O3_vasprun.xml.relax2.gz')).get_computed_entry()

        if not stable_bio_entry:
            stable_bio_df = pd.DataFrame.from_csv(os.path.join(MODULE_DIR, 'tools/stable_binary_oxides_garnet.csv'))
            stable_bio_id = stable_bio_df.loc[lambda df: df.specie == el.symbol]['mpid'].tolist()[0]
            stable_bio_entry = m.get_entry_by_material_id(stable_bio_id,
                                                          property_data=['e_above_hull', 'formation_energy_per_atom'])
        min_e = stable_bio_entry.uncorrected_energy
        amt_bio = stable_bio_entry.composition[el.name]
        bio_form_e -= (amt / (amt_bio)) * min_e

    f = entry.composition.num_atoms if factor == 'per_atom' else entry.composition.get_integer_formula_and_factor()[1]

    return bio_form_e / f


def get_form_e_from_bio_perov(entry_dict, factor='per_fu', energy=None, spes=None):
    """
    calculate formation energy for single perovskites
    :param entry_dict: dict
        vasprun.get_computed_entry().as_dict()
    :param factor: str 'per_fu' or 'per_atom'
        unit of formation energy
    :param energy: float
        total energy of the entry, if not given ,use entry.uncorrected_energy
    :param spes: list
        list of speces ['Ba2+','Ti4+']

    :return: float
        formation energy from binary oxides
    """
    """
    arg:
    entry_dict :
    factor:
    return:
    """
    entry = ComputedEntry.from_dict(entry_dict)
    if energy:
        bio_form_e = energy
    else:
        bio_form_e = entry.uncorrected_energy

    m = MPRester()

    for el in spes:

        stable_bio_entry = None
        if el == 'O2-':
            continue
        el_name = re.split('[^a-zA-Z]', el)[0]
        amt = entry.composition[el_name]
        if el == 'Yb3+':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/mp-2814_Yb16O24_vasprun.xml.relax2.gz')).get_computed_entry()
        if el == 'Nb5+':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/ICSD_25750_Nb2O5_vasprun.xml.relax2.gz')).get_computed_entry()
        if el == 'Eu3+':
            stable_bio_entry = Vasprun(
                os.path.join(MODULE_DIR, 'data/ICSD_40472_Eu2O3_vasprun.xml.relax2.gz')).get_computed_entry()

        if not stable_bio_entry:
            stable_bio_df = pd.DataFrame.from_csv(os.path.join(MODULE_DIR, 'tools/stable_binary_oxides_perov.csv'))
            stable_bio_id = stable_bio_df.loc[lambda df: df.specie == el]['mpid'].tolist()[0]
            stable_bio_entry = m.get_entry_by_material_id(stable_bio_id,
                                                          property_data=['e_above_hull', 'formation_energy_per_atom'])
        min_e = stable_bio_entry.uncorrected_energy
        amt_bio = stable_bio_entry.composition[el_name]
        # print(bio_form_e,stable_bio_id,stable_bio_entry.composition,amt_bio,amt/amt_bio,min_e,stable_bio_entry.energy_per_atom)
        bio_form_e -= (amt / (amt_bio)) * min_e
        # print(bio_form_e,stable_bio_entry.entry_id )

    f = entry.composition.num_atoms if factor == 'per_atom' else entry.composition.get_integer_formula_and_factor()[1]

    return bio_form_e / f
