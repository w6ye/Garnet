import unittest
from garnet.bandgap.data import load_unmix_data
import numpy as np
from pymatgen.core.periodic_table import get_el_sp

def pow2(x):
    return pow(x, 2)

def ln(x):
    return np.log(1+x)

class TestLoadUnmixData(unittest.TestCase):

    def testbasic(self):
        properties = ['Ionic radius', 'Pauling EN']
        spes_ss, in_ss, bandgap_ss = load_unmix_data(properties=properties,
                                                     filter={"formula": "Mg12Ga8Ti12O48"})
        spes_ss_ans = {"c": {"Mg2+": 3},
                       "a": {"Ga3+": 2},
                       "d": {"Ti4+": 3}}
        bandgap_ss_ans = 4.0052
        # eles = map(get_el_sp, ["Mg2+", "Ga3+", "Ti4+"])
        # in_ss_ans = [(getattr(el, "average_ionic_radius"), getattr(el, "X"))\
        #         for el in eles]
        # in_ss_ans = list(sum(in_ss_ans,()))
        in_ss_ans = [0.86, 1.31, 0.76, 1.81, 0.8516666666666667, 1.54]

        self.assertEqual(spes_ss[0], spes_ss_ans)
        self.assertEqual(bandgap_ss[0], bandgap_ss_ans)
        self.assertEqual(in_ss[0], in_ss_ans)

    def testvariantion(self):
        self.maxDiff = None
        properties = ['Ionic radius', 'Pauling EN']
        variations = [lambda x: x, pow2, np.sqrt, np.cbrt, np.exp, ln]
        _, in_ss, _ = load_unmix_data(properties=properties,
                                      filter={"formula":
                                             "Mg12Ga8Ti12O48"},
                                      variations=variations)
        in_ss_ans = [0.86, 1.31,
                     pow2(0.86), pow2(1.31),
                     np.sqrt(0.86), np.sqrt(1.31),
                     np.cbrt(0.86),np.cbrt(1.31),
                     np.exp(0.86),np.exp(1.31),
                     ln(0.86), ln(1.31),
                     0.76, 1.81,
                     pow2(0.76), pow2(1.81),
                     np.sqrt(0.76), np.sqrt(1.81),
                     np.cbrt(0.76), np.cbrt(1.81),
                     np.exp(0.76), np.exp(1.81),
                     ln(0.76), ln(1.81),
                     0.8516666666666667, 1.54,
                     pow2(0.8516666666666667), pow2(1.54),
                     np.sqrt(0.8516666666666667), np.sqrt(1.54),
                     np.cbrt(0.8516666666666667), np.cbrt(1.54),
                     np.exp(0.8516666666666667), np.exp(1.54),
                     ln(0.8516666666666667), ln(1.54),
                     ]
        self.assertEqual(in_ss[0], in_ss_ans)




if __name__ == "__main__":
    unittest.main()
