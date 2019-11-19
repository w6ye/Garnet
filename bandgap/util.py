import numpy as np
from pymatgen import Composition

STD_FORMULA = {'garnet': Composition("C3A2D3O12"),
               "perovskite": Composition("A2B2O6")}
SITES = {'garnet': ['c', 'a', 'd'],
         'perovskite': ['a', 'b']}  # use list to preserve order
SITE_INFO = {'garnet': {'c': {"num_atoms": 3, "max_ordering": 20, "cn": "VIII"},
                        'a': {"num_atoms": 2, "max_ordering": 7, "cn": "VI"},
                        'd': {"num_atoms": 3, "max_ordering": 18, "cn": "IV"}},
             'perovskite': {'a': {"num_atoms": 2, "max_ordering": 10, 'cn': "XII"},
                            'b': {"num_atoms": 2, "max_ordering": 10, 'cn': "VI"}}}



def spe2form(structure_type, species):
    """
    Transfer from a given species dict to the
    standard perovskite formula. (A2B2O6)

    Args:
        structure_type (str): garnet or perovskite
        species (dict): species in dictionary.
            e.g. for Ca2Ti2O6,
                species = {
                            "a": {"Ca2+": 1},
                            "b": {"Ti4+": 1}}
            e.g. for CaSrTi2O6:
                species = {"a": {"Ca2+":0.5,
                                "Sr2+": 0.5},
                           "b": {"Ti4+": 1}}
    Returns:
        formula (str)
    """
    sites = SITES[structure_type]

    spe_list = [spe.name + str(round(SITE_INFO[structure_type][site]['num_atoms'] \
                                     * species[site][spe]))
                for site in sites for
                spe in sorted(species[site], key=lambda x: species[site][x])]
    formula = "".join(spe_list)
    num_oxy = int(STD_FORMULA[structure_type]['O'])
    formula = formula.replace("1", "") + 'O%s' % num_oxy
    return formula

class log_scaler():
    """
    This class is used to log scale the y data
    """

    def __init__(self, e=2):
        self.e = e

    def transform(self, y):
        return np.log(np.array(y) + self.e)

    def inverse(self, y_scaled):
        return np.exp(np.array(y_scaled)) - self.e


def log_error(y, y_pred, e=2):
    """
    score function to be used in GridSearchCV
    Example:
        scorer = make_scorer(log_error, greater_is_better=False)
        GridSearchCV(scoring=scorer, **kwrds)
    """
    return np.sum((np.log(e + y_pred) - np.log(e + y)) ** 2, axis=-1)


def get_mae(grid_cv, x, y_true, yscaler):
    """
    Get mae of data
    (The score out of GridSearchCV is not in mae if used customized scorer)
    """
    y_pred = grid_cv.best_estimator_.predict(x)
    y_pred_inv = yscaler.inverse(y_pred)
    return np.mean(np.abs(y_pred_inv - yscaler.inverse(y_true)))

