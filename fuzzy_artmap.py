import numpy as np

class FuzzyARTMAP:
    """
    Fuzzy ARTMAP
    Entrada: vetores contínuos x (0 a 1) + rótulo y (0 a 1)
    Objetivo: classificação supervisionada com entradas contínuas
    """

    def __init__(self, n_a, n_b, vigilancia=0.75, beta=1.0,
                 alpha=0.001, epsilon=0.001):
        self.n_a    = n_a           # features de x
        self.n_b    = n_b           # tamanho do rótulo y
        self.rho_a  = vigilancia    # vigilância base
        self.beta   = beta          # taxa de aprendizado
        self.alpha  = alpha         # evita divisão por zero
        self.eps    = epsilon       # incremento Match Tracking
        self.W_a    = []            # pesos Fuzzy ART-a (tamanho 2*n_a)
        self.W_b    = []            # pesos Fuzzy ART-b (tamanho 2*n_b)
        self.mapa   = {}            # mapa j → k

    # ----------------------------------------------------------
    def _comp(self, x):
        """Complementa: [x, 1-x]"""
        return np.concatenate([x, 1 - x])

    # ----------------------------------------------------------
    def _ativacao_fuzzy(self, x_comp, W):
        """T(j) = |min(x_comp, w(j))| / (α + |w(j)|) para todos j"""
        T = []
        for w in W:
            minimo = np.minimum(x_comp, w)
            T.append(minimo.sum() / (self.alpha + w.sum()))
        return np.array(T)

    # ----------------------------------------------------------
    def _vigilancia_ok(self, x_comp, w, rho):
        minimo  = np.minimum(x_comp, w)
        soma_x  = x_comp.sum()
        if soma_x == 0:
            return False
        return (minimo.sum() / soma_x) >= rho

    # ----------------------------------------------------------
    def _atualizar(self, x_comp, W, j):
        """w(j) = β*min(x_comp, w(j)) + (1-β)*w(j)"""
        minimo = np.minimum(x_comp, W[j])
        W[j]   = self.beta * minimo + (1 - self.beta) * W[j]

    # ----------------------------------------------------------
    def _melhor_b(self, y_comp):
        """Encontra categoria k* em Fuzzy ART-b"""
        if len(self.W_b) == 0:
            self.W_b.append(np.ones(2 * self.n_b))
            return 0
        T_b = self._ativacao_fuzzy(y_comp, self.W_b)
        return int(np.argmax(T_b))

    # ----------------------------------------------------------
    def treinar(self, x, y):
        """Treina com par (x, y) e retorna (j*, k*)"""
        x      = np.array(x, dtype=float)
        y      = np.array(y, dtype=float)
        x_comp = self._comp(x)
        y_comp = self._comp(y)

        # Melhor categoria em Fuzzy ART-b
        k_star = self._melhor_b(y_comp)

        # Inicializa Fuzzy ART-a se necessário
        if len(self.W_a) == 0:
            self.W_a.append(np.ones(2 * self.n_a))

        T_a       = self._ativacao_fuzzy(x_comp, self.W_a)
        ordem     = np.argsort(T_a)[::-1]
        rho_atual = self.rho_a

        for j_star in ordem:
            if self._vigilancia_ok(x_comp, self.W_a[j_star], rho_atual):

                if j_star not in self.mapa or self.mapa[j_star] == k_star:
                    # RESSONÂNCIA
                    self._atualizar(x_comp, self.W_a, j_star)
                    self._atualizar(y_comp, self.W_b, k_star)
                    self.mapa[j_star] = k_star
                    return int(j_star), int(k_star)

                else:
                    # MATCH TRACKING
                    minimo    = np.minimum(x_comp, self.W_a[j_star])
                    rho_atual = (minimo.sum() / x_comp.sum()) + self.eps

        # Nenhuma passou — cria nova categoria
        self.W_a.append(np.ones(2 * self.n_a))
        j_novo = len(self.W_a) - 1
        self._atualizar(x_comp, self.W_a, j_novo)
        self._atualizar(y_comp, self.W_b, k_star)
        self.mapa[j_novo] = k_star
        return int(j_novo), int(k_star)

    # ----------------------------------------------------------
    def classificar(self, x):
        """Classifica x sem atualizar pesos"""
        x      = np.array(x, dtype=float)
        x_comp = self._comp(x)
        T_a    = self._ativacao_fuzzy(x_comp, self.W_a)
        for j in np.argsort(T_a)[::-1]:
            if self._vigilancia_ok(x_comp, self.W_a[j], self.rho_a):
                if j in self.mapa:
                    return self.mapa[j]
        return -1


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":

    # Dados contínuos com rótulo one-hot
    dados = [
        ([0.9, 0.1, 0.0, 0.1], [1.0, 0.0]),   # classe 0
        ([0.8, 0.2, 0.1, 0.0], [1.0, 0.0]),   # classe 0
        ([0.0, 0.9, 0.8, 0.9], [0.0, 1.0]),   # classe 1
        ([0.1, 0.8, 0.9, 0.8], [0.0, 1.0]),   # classe 1
        ([0.7, 0.1, 0.0, 0.2], [1.0, 0.0]),   # classe 0
        ([0.0, 0.7, 0.9, 0.7], [0.0, 1.0]),   # classe 1
    ]

    fartmap = FuzzyARTMAP(n_a=4, n_b=2, vigilancia=0.6)

    print("=== Fuzzy ARTMAP — Treinamento ===")
    for i, (x, y) in enumerate(dados):
        j, k = fartmap.treinar(x, y)
        print(f"  Exemplo {i+1}: x={x} y={y} → Cat_a={j} Cat_b={k}")

    print("\n=== Fuzzy ARTMAP — Classificação ===")
    testes = [
        [0.85, 0.15, 0.0, 0.1],   # deve ser classe 0
        [0.05, 0.75, 0.85, 0.8],  # deve ser classe 1
    ]
    for x in testes:
        k = fartmap.classificar(x)
        print(f"  x={x} → Classe {k}")