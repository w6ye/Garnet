


from pymacy.icsd.db import get_db
from pymatgen import Structure
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.io.vasp import Vasprun
from pymatgen.phasediagram.maker import PhaseDiagram
from pymatgen.phasediagram.analyzer import PDAnalyzer
from pymacy.qe import MVLQE
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen import MPRester
from proj_garnet.garnet.util import get_site_spe_from_structure,get_averaged_input
from proj_garnet.garnet.analysis import get_ehull,get_form_e_from_bio
import pandas as pd
import os
import logging
import json
import warnings


log = logging.getLogger()

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(MODULE_DIR, '../tools/elements.json'), 'r') as f:
    garnet_elements = json.load(f)
a = MPRester(api_key='FI7XgMqWtTkEcDUC')
warnings.filterwarnings('ignore')


# def get_form_e_from_bio(entry_dict,factor='per_fu',energy=None):
#     """
#     arg: entry_dict : vasprun.get_computed_entry().as_dict()
#     return: formation energy from binary oxides (eV/atom)
#     """
#     entry = ComputedEntry.from_dict(entry_dict)
#     if energy:
#        bio_form_e = energy
#     else:
#        bio_form_e = entry.uncorrected_energy
#
#     m = MPRester()
#
#
#     for el,amt in entry.composition.items():
#         stable_bio_entry = None
#         if el.symbol == 'O':
#             continue
#         if el.symbol == 'Yb':
#
#             stable_bio_entry = Vasprun(os.path.join(MODULE_DIR,'../data/mp-2814_Yb16O24_vasprun.xml.relax2.gz')).get_computed_entry()
#         if el.symbol == 'Nb':
#             stable_bio_entry = Vasprun(os.path.join(MODULE_DIR, '../data/ICSD_25750_Nb2O5_vasprun.xml.relax2.gz')).get_computed_entry()
#         if el.symbol == 'Eu':
#             stable_bio_entry = Vasprun(os.path.join(MODULE_DIR, '../data/ICSD_40472_Eu2O3_vasprun.xml.relax2.gz')).get_computed_entry()
#
#         if not stable_bio_entry:
#             stable_bio_df = pd.DataFrame.from_csv(os.path.join(MODULE_DIR,'../tools/stable_binary_oxides_garnet.csv'))
#             stable_bio_id = stable_bio_df.loc[lambda df: df.specie==el.symbol]['mpid'].tolist()[0]
#             stable_bio_entry = m.get_entry_by_material_id(stable_bio_id,property_data=['e_above_hull', 'formation_energy_per_atom'])
#         min_e = stable_bio_entry.uncorrected_energy
#         amt_bio = stable_bio_entry.composition[el.name]
#         bio_form_e -= (amt/(amt_bio)) * min_e
#
#
#     f = entry.composition.num_atoms if factor == 'per_atom' else entry.composition.get_integer_formula_and_factor()[1]
#
#     return bio_form_e/f





def get_raw_entries_all(root,data_type,outdir=None):

    entries = []

    for m,n,k in os.walk(root):
        for folder in n:
            if folder == 'relax':
                if data_type == 'sub':
                    formula = m.split('/')[-1]
                    site = 0
                elif data_type == 'mix':
                    formula = m.split('/')[-2]
                    site = m.split('/')[-1]

                vasp = os.path.join(m,folder,'vasprun.xml.relax2.gz')
                if not os.path.isfile(vasp):
                    continue
                try:
                    v = Vasprun(vasp)
                    if v.converged:
                        s = v.final_structure
                        entry = v.get_computed_entry()
                except:
                    log.info("%s_%s not done"%(formula,site))
                    continue
                entries.append({'formula':formula,'Site':site,'Etot':v.final_energy,'structure':s.as_dict(),'entry':entry.as_dict()})
    if outdir:
        with open('%s.json' % outdir, 'w') as f:
            json.dump(entries, f)

    return entries

def get_raw_entry_one(vasprun,formula,site):
    """

    :param vasprun: pymatgen.Vasprun file
    :param formula: in C3A2D3O12 format
    :param site: 0 for all non-mixed
    :param data_type: mixed or non_mixed

    :return:
    """


    raw_entry = {}
    if vasprun.converged:

        s = vasprun.final_structure
        entry = vasprun.get_computed_entry()
    else:
        log.info("%s not done"%formula)
        return None


    raw_entry.update({'formula':formula,'Site':site,'Etot':vasprun.final_energy,'structure':s.as_dict(),'entry':entry.as_dict()})

    return raw_entry



def get_garnet_ml_entry_one(raw_entry):

    s = Structure.from_dict(raw_entry['structure'])
    entry = raw_entry['entry']
    site_spes = get_site_spe_from_structure(s)
    a = site_spes['a']
    d = site_spes['d']
    c = site_spes['c']
    elements = list(set([site.specie.name for site in s]))
    s_ox = s.add_oxidation_state_by_element(garnet_elements)
    ion_spes = [site.specie.__str__() for site in s_ox]
    Xs = {}
    Irs = {}
    for ion in ion_spes:
        Xs.update({ion:get_el_sp(ion).X})
        Irs.update({ion:get_el_sp(ion).ionic_radius})
    space_group = {'symbol':s.get_space_group_info()[0],'number':s.get_space_group_info()[1]}
    averaged_input = get_averaged_input(site_spes)
    form_e_from_bi = get_form_e_from_bio(entry_dict=entry)
    ehull = get_ehull(entry_dict = entry,dbpath = os.path.join(MODULE_DIR,'../tools/cal_db.json'),corrected=False)

    garnet_entry = {
        'structure' : s.as_dict(),
        'a': a,
        'd': d,
        'c': c,
        'elements': elements,
        'nelements': len(elements),
        'nelements_a': len(a),
        'nelements_d': len(d),
        'nelements_c': len(c),
        'electronegtivity': Xs,
        'ionic_radius': Irs,
        'species': ion_spes,
        'spacegroup': space_group,
        'averaged_input': averaged_input,
        'form_e_from_bi': form_e_from_bi,
        'ehull': ehull
        }
    return garnet_entry





