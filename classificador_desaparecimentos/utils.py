import re
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def clean_text(x):
    return re.sub('[^a-zA-Z ]', '', unidecode(x).upper())

# TODO: See how to pass kwargs to the model itself
# TODO: Delete no apparent reason before? Or add argument to say which column indicates no reason (seems best)
class OneVsRestLogisticRegression:
    def __init__(self, **kwargs):
        self.model_ = OneVsRestClassifier(LogisticRegression(**kwargs))

    def fit(self, X, y):
        self.model_.fit(X, y)

    def predict(self, X):
        p = self.model_.predict(X)
        p = np.insert(p, 12, values=(p.sum(axis=1) == 0).astype(int), axis=1)
        return p

class RegexClassifier:
    def __init__(self, rules, name_to_key_dict=None):
        self.rules = deepcopy(rules)
        if name_to_key_dict:
            names = list(self.rules.keys())
            for name in names:
                self.rules[int(name_to_key_dict[name])] = self.rules.pop(name)
    
    def predict(self, texts):
        results = []
        for t in texts:
            t_classes = []
            for c in self.rules:
                for expression in self.rules[c]:
                    m = re.search(expression, unidecode(t).upper())
                    if m:
                        t_classes.append(c)
                        break
            results.append(t_classes)
        return results
    
rules = {
    'DROGADIÇÃO': ['USUARI[OA] DE (DROGA|ENTORPECENTE)S?', 'ALCOOLATRA', 'COCAINA', 'VICIAD[OA]', 'DEPENDENTE QUIMICO', 'MACONHA', 'ALCOOL', 'CRACK'],
    'PROBLEMAS PSIQUIÁTRICOS': ['DEPRESSAO', 'ESQUI[ZS]OFRENIA', 'ESQUI[ZS]OFRENIC[OA]', 'ALZHEIMER', 
                                '(DOENCA|TRANSTORNO|PROBLEMA|DISTURBIO)S? MENTA(L|IS)'],
    #'CONFLITO INTRAFAMILIAR': ['DESENTENDIMENTO', ' BRIGA', ' BRIGOU', 'SE DESENTENDEU', 'DISCUTIU', 'DISCUTIRAM', 'DISCUSSAO'],
    'ENVOLVIMENTO COM TRÁFICO DE ENTORPECENTES': [' TRAFICO', 'TRAFICANTES'],
    'PRISÃO/APREENSÃO': ['ABORDAD[OA] (POR POLICIAIS|PELA POLICIA)'],
    'CATÁSTROFE': ['FORTES CHUVAS', 'TEMPESTADE', 'ENXURRADA', 'DESLIZAMENTO', 'ROMPIMENTO D[EA] BARRAGEM', 'SOTERRAD[OA]'],
    'POSSÍVEL VÍTMA DE SEQUESTRO': [' RAPTOU ', ' RAPTAD[OA] ', 'SEQUESTROU?', 'SEQUESTRAD[OA]']
}