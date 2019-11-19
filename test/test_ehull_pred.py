from __future__ import division, print_function, unicode_literals, \
    absolute_import

import unittest
from pymatgen.core import Structure
from pymatgen.core.periodic_table import get_el_sp
from garnet.ehull_pred import spe2form, get_input,get_form_e,get_tote,get_ehull
from keras.models import load_model
import os
import pickle


class GarnetEhullPredTest(unittest.TestCase):
    def setUp(self):
        keys = ["a_eneg", "a_radius", "c_eneg", "c_radius", "d_eneg", "d_radius"]
        self.spes = {"c": {"Zr4+": 1, "Zn2+": 2},
                     "a": {"Mg2+": 2},
                     "d": {"Si4+": 3}}
        self.average_inputs = {"a_eneg": 1.31,
                               "a_radius": 0.86,
                               "c_eneg": 1.5373439617208791,
                               "c_radius": 0.8733333333333333,
                               "d_eneg": 1.9,
                               "d_radius": 0.54}

        self.inputs_n = [self.average_inputs[key] for key in keys]
        model_path = "../../data/models/2017-10-11/model_6_24_1_all_epoch500.h5"
        scaler_path = "../../data/models/2017-10-11/scaler_epoch500.pkl"
        self.model = load_model(model_path)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        inputs_ns = self.scaler.transform(self.inputs_n)
        self.predicted_form_e = self.model.predict(inputs_ns.reshape(1, 6))[0][0]



    def test_spe2form(self):
        self.assertTrue(spe2form(self.spes),"ZrZn2Mg2Si4O12")

    def test_form_e(self):
        inputs = get_input(self.spes)
        self.assertTrue(inputs, self.average_inputs)
        form_e = get_form_e(inputs, self.model, self.scaler)
        self.assertTrue(form_e,1.55637658)

    def test_tote(self):
        tote = get_tote(self.predicted_form_e, self.spes)

    def test_ehull(self):
        y3al5o12_tote = -651.5696
        species = {"c":{"Y":12},"a":{"Al":8},"d":{"Al":12}}
        y3al5o12_ehull = get_ehull(y3al5o12_tote, species=species)
        self.assertAlmostEqual(y3al5o12_ehull,0)

        cr2o3_tote =  -316.2270
        cr2o3_formula = 'Cr16O24'
        cr2o3_ehull = get_ehull(cr2o3_tote, formula=cr2o3_formula)
        self.assertAlmostEqual(round(cr2o3_ehull,3), 0.033)




if __name__ == "__main__":
    unittest.main()