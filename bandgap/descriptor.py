import os
from monty.serialization import loadfn
from pymatgen.core.periodic_table import get_el_sp
import numpy as np

module_path = os.path.dirname(os.path.abspath(__file__))

IE = loadfn(os.path.join(module_path, "../../tools/IE.json"))
EA = loadfn(os.path.join(module_path, "../../tools/EA.json"))
Valence = loadfn(os.path.join(module_path, "../../tools/valence.json"))

pa_dict = {'Boiling point': "boiling_point",
           'Mendeleev number': "mendeleev_no",
           'Ionic radius': "average_ionic_radius",
           'Group number': "group",
           #  'Covalent radius',
           'First ionization energy': "",
           'Density': "density_of_solid",
           'No. valence electrons': "",
           'Melting point': "melting_point",
           'Atomic mass': "atomic_mass",
           'Electron affinity': "",
           'Atomic radius': "atomic_radius",
           'Period number': "row",
           'Atomic number': "number",
           'Pauling EN': "X"}


def get_lowest_config(entry):
    """
    Args:
    :param entry:
    :return:
    """
    calcs = sorted(entry['calculations'], key=lambda x: entry['calculations'][x]['energy'])
    return calcs[0]


def get_X_mixed(spes):
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

    num_sites = sum([v for k, v in spes.items()])
    if type(spes) == str:
        el = get_el_sp(spes)
        return el.X
    elif type(spes) == dict:
        avg_eneg = 0

        for s, amt in spes.items():
            el = get_el_sp(s)
            avg_eneg += (amt / num_sites) * (el.X - o.X) ** 2

        return np.abs(o.X - (np.sqrt(avg_eneg)))


def get_ie_mixed(spes, method='max'):
    """
    :param spe: str or dict
        element in str or a dictionary, the oxistate is defined in garnet elements unless otherwise specified,

    :return: float
        ionic radius, if spe is in dict, return weighted mean ir
    """

    if type(spes) == str:
        return IE[get_el_sp(spes.element.name)['CRC']['1']]

    if type(spes) == dict and method == 'avg':
        mean_ie = 0

        factor = sum([spes[el] for el in spes])

        for el in spes:
            mean_ie += IE[get_el_sp(el).element.name]['CRC']['1'] * spes[el] / factor
        return mean_ie

    elif type(spes) == dict and method == 'max':
        ies = [IE[get_el_sp(el).element.name]['CRC']['1'] for el in spes]
        return max(ies)


def get_ea_mixed(spes, method='max'):
    if type(spes) == str:
        return EA[get_el_sp(spes.element.name)]

    if type(spes) == dict and method == 'avg':
        mean_ea = 0

        factor = sum([spes[el] for el in spes])
        for el in spes:
            mean_ea += EA[get_el_sp(el).element.name] * spes[el] / factor
        return mean_ea

    elif type(spes) == dict and method == 'max':
        eas = [EA[get_el_sp(el).element.name] for el in spes]
        return max(eas)


def get_valence_mixed(spes):
    if type(spes) == str:
        return float(Valence[get_el_sp(spes.element.name)]['zval'])

    if type(spes) == dict:
        mean_ea = 0

        factor = sum([spes[el] for el in spes])
        for el in spes:
            mean_ea += float(Valence[get_el_sp(el).element.name]['zval']) \
                       * spes[el] / factor
        return mean_ea


def get_average_descriptors(spes, p, method='max'):
    if p == 'Electron affinity':
        return get_ea_mixed(spes, method)
    elif p == 'First ionization energy':
        return get_ie_mixed(spes, method)
    elif p == 'No. valence electrons':
        return get_valence_mixed(spes)
    elif p == 'Pauling EN':
        return get_X_mixed(spes)
    elif type(spes) == str:
        return getattr(get_el_sp(spes).element, pa_dict[p])

    elif type(spes) == dict:
        mean_pro = 0
        factor = sum([spes[el] for el in spes])
        for el in spes:
            mean_pro += getattr(get_el_sp(el).element, pa_dict[p]) * spes[el] / factor
        return mean_pro

