from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
from pymatgen.core import Structure
from pymatgen.core.periodic_table import get_el_sp
import os

from garnet.util import *


class GarnetUtilTest(unittest.TestCase):
    def setUp(self):
        self.icsd1 = 170157
        self.icsd2 = 63650
        self.y3al5o12 = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                         "Y3Al5O12.cif"))
        self.basg = Structure.from_file(os.path.join(os.path.dirname(__file__),
                                                     "Ba3Al2Ge2SiO12.cif"))
        self.site_spe_basg = {'a': {'Al': 2.0}, 'c': {'Ba': 3.0}, 'd': {'Ge': 2.0, 'Si': 1.0}}
        self.y3al5o12.add_oxidation_state_by_element({"Y": +3, "Al": +3, "O": -2})
        self.basg.add_oxidation_state_by_element({"Ba": +2, "Al": +3, "Si": +4, "Ge": +4, "O": -2})
        self.o = get_el_sp('O2-')


    def test_mpid(self):
        mpid1 = get_mpid(self.icsd1)
        mpid2 = get_mpid(self.icsd2)
        self.assertEqual(mpid1,'mp-3050')
        self.assertEqual(mpid2,'mp-5409')


    def test_get_site_spe(self):
        site_spe_yag = get_site_spe_from_structure(self.y3al5o12)
        site_spe_basg = get_site_spe_from_structure(self.basg)
        site_spe_ggg = get_site_spe_from_formula("Ga3Gd5O12")
        site_spe_gsa = get_site_spe_from_formula("Gd3Sc2Ga3O12")
        site_spe_bags2 = get_site_spe_from_formula("Ba3Al2Ge2SiO12")
        self.assertEqual(site_spe_yag,{'a': {'Al': 2.0}, 'c': {'Y': 3.0}, 'd': {'Al': 3.0}})
        self.assertEqual(site_spe_basg,{'a': {'Al': 2.0}, 'c': {'Ba': 3.0}, 'd': {'Ge': 2.0, 'Si': 1.0}})
        self.assertEqual(site_spe_ggg,{'a': {'Gd': 2}, 'c': {'Ga': 3}, 'd': {'Gd': 3}})
        self.assertEqual(site_spe_gsa,{'a': {'Sc': 2}, 'c': {'Gd': 3}, 'd': {'Ga': 3}})
        self.assertEqual(site_spe_bags2,{'a': {'Al': 2}, 'c': {'Ba': 3}, 'd': {'Ge': 2, 'Si': 1}})

    def test_input(self):
        Ag_ir = get_ir_one('Ag')
        gesi_ir = get_ir_mixed({'Ge': 2.0, 'Si': 1.0})
        Ag_x = get_X_one('Ag')
        gesi_X = get_X_BD_mixed({'Ge': 2.0, 'Si': 1.0})
        self.assertEqual(Ag_ir,1.29)
        self.assertEqual(gesi_ir,((0.67*2)+(0.54*1))/3)
        self.assertEqual(Ag_x,1.93)
        ge = get_el_sp('Ge4+')
        si = get_el_sp('Si4+')
        self.assertEqual(gesi_X,1.972416952946103)

        input_basg = get_averaged_input(self.site_spe_basg)
        self.assertEqual(input_basg,{"a_eneg":1.61,
                                     "a_radius":0.675,
                                     "c_eneg":0.89,
                                     "c_radius":1.49,
                                     "d_eneg":1.972416952946103,
                                     "d_radius":0.6266666666666667})






if __name__ == "__main__":
    unittest.main()





