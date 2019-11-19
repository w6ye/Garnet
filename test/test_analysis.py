import unittest
from pymatgen.io.vasp import Vasprun
from pymatgen import MPRester
from garnet.analysis import get_ehull, get_form_e_from_bio, get_pred_ehull
from garnet.util import get_averaged_input

from keras.models import load_model
import pickle


class GarnetAnalysisTest(unittest.TestCase):
    def setUp(self):
        m = MPRester()
        self.yag_entry = m.get_entry_by_material_id('Y3Al5O12')
        self.maso_entry = m.get_entry_by_material_id('Mg3Al2(SiO4)3')

        self.model = load_model("model_25.h5")
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.yag_input = get_averaged_input({'a': {'Al': 2.0}, 'c': {'Y': 3.0}, 'd': {'Al': 3.0}})
        self.input_keys = ['a_radius','a_eneg',  'c_radius', 'c_eneg', 'd_radius','d_eneg']
        self.yag_input_nu = [self.yag_input[key] for key in self.input_keys ]
        self.yag_vasprun = Vasprun('yag_vasprun.xml.relax2.gz')

        # entry for Li2SiO3
        self.lso = m.get_entry_by_material_id("mp-16626")
        # entry for SrSiO3
        self.sso = m.get_entry_by_material_id('mp-3978')
        # entry for AlLaO3
        self.alo = m.get_entry_by_material_id("mp-2920")

    def test_ehull(self):
        ehull_yag = get_ehull(self.yag_entry.as_dict())
        ehull_maso = get_ehull(self.maso_entry.as_dict())

        self.assertEqual(ehull_yag, 0)
        self.assertEqual(round(ehull_maso, 3), 0.039)

    def test_bio_form_e(self):
        lso_bio_e = get_form_e_from_bio(self.lso.as_dict(), factor='per_atom')
        alo_bio_e = get_form_e_from_bio(self.alo.as_dict(), factor='per_atom')
        sso_bio_e = get_form_e_from_bio(self.sso.as_dict(), factor='per_atom')
        sso_bio_e_fu = get_form_e_from_bio(self.sso.as_dict())
        self.assertEqual(sso_bio_e_fu / sso_bio_e, 5)
        self.assertTrue(abs(sso_bio_e - (-0.257)) < 1e-3)
        #self.assertTrue(abs(lso_bio_e - (-0.225)) < 2 * 1e-3)
        self.assertTrue(abs(alo_bio_e - (-0.072)) < 1e-3)

    def test_predict_ehull(self):
        yag_bio_e = get_form_e_from_bio(self.yag_vasprun.get_computed_entry().as_dict())
        yag_input_scaled = self.scaler.fit_transform(self.yag_input_nu)
        yag_bio_e_pred = self.model.predict(yag_input_scaled.reshape(1,6)).tolist()[0][0]
        yag_tote_calc = self.yag_vasprun.final_energy
        yag_tote_pre, yag_ehull_pred = get_pred_ehull(tote_calc=yag_tote_calc,
                                                      form_e_calc=yag_bio_e*4,
                                                      form_e_predict=yag_bio_e_pred*4,
                                                      elements=["Y", "Al", "O"],
                                                      composition="Y3Al5O12")
        self.assertAlmostEqual((yag_tote_pre - yag_tote_calc),(yag_bio_e_pred - yag_bio_e))
        self.assertEqual(yag_ehull_pred, 0)


if __name__ == "__main__":
    unittest.main()
