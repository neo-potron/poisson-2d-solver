"""
poisson.py — Noyau physique et algébrique du solveur de Poisson 2D.

Ce module fournit :
    - setup_charges()       : initialisation de la distribution de charges ρ(x,y)
    - setup_permittivity()  : initialisation de la carte de permittivité relative ε_r(x,y)
    - Poisson2D             : classe solveur encapsulant la construction de la matrice
                              du système et six algorithmes de résolution distincts.

L'équation résolue est la forme généralisée de Poisson :
    -∇·[ε_r(x,y) ∇V(x,y)] = ρ(x,y) / ε_0   (en unités adimensionnelles ε_0 = 1)
avec conditions aux limites de Dirichlet homogènes (V = 0 sur ∂Ω).

Auteurs : Néo Potron & Alex Cassi
Date    : 24/04/2026
Cours   : Méthodes Numériques
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Fonctions de configuration physique
# =============================================================================

def setup_charges(I: int, J: int, Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    """
    Initialise la distribution de densité de charge surfacique ρ(x,y).

    La distribution modélise un condensateur plan idéal orienté selon Y :
        - Ligne de charges positives  (+1) centrée en y = 0.4
        - Ligne de charges négatives  (-1) centrée en y = 0.6
    Les deux lignes s'étendent en x sur [0.25, 0.75].
    Les nœuds de bord (conditions de Dirichlet) sont laissés à zéro.

    IN:
        I   (int)   : Nombre de nœuds selon x.
        J   (int)   : Nombre de nœuds selon y.
        Lx  (float) : Longueur physique du domaine selon x (défaut 1.0).
        Ly  (float) : Longueur physique du domaine selon y (défaut 1.0).
    OUT:
        rho_vector (np.ndarray) : Vecteur aplati de taille IxJ contenant ρ.
    """
    rho = np.zeros((I, J))
    dx, dy = Lx / (I - 1), Ly / (J - 1)

    for i in range(I):
        x = i * dx
        for j in range(J):
            y = j * dy
            # Zone active : x ∈ [0.25, 0.75]
            if 0.25 <= x <= 0.75:
                if   abs(y - 0.4) < dy / 2: rho[i, j] = 1.0   # Armature positive
                elif abs(y - 0.6) < dy / 2: rho[i, j] = -1.0   # Armature négative

    # Aplatissement en vecteur colonne avec zeroing des bords (Dirichlet V=0)
    rho_vector = np.zeros(I * J)
    for i in range(I):
        for j in range(J):
            k = i * J + j
            if not (i == 0 or i == I - 1 or j == 0 or j == J - 1):
                rho_vector[k] = rho[i, j]

    return rho_vector


def setup_permittivity(I: int, J: int, Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    """
    Initialise la carte de permittivité relative ε_r(x,y).

    Modélise un bloc diélectrique rectangulaire de permittivité ε_r = 50
    centré dans le domaine, simulant par exemple un matériau céramique :
        - x ∈ [0.1, 0.5]
        - y ∈ [0.45, 0.55]
    Le reste du domaine est du vide (ε_r = 1).

    IN:
        I   (int)   : Nombre de nœuds selon x.
        J   (int)   : Nombre de nœuds selon y.
        Lx  (float) : Longueur physique du domaine selon x (défaut 1.0).
        Ly  (float) : Longueur physique du domaine selon y (défaut 1.0).
    OUT:
        eps_r (np.ndarray) : Matrice (IxJ) de permittivités relatives.
    """
    eps_r = np.ones((I, J))   # Vide partout par défaut
    dx, dy = Lx / (I - 1), Ly / (J - 1)

    for i in range(I):
        x = i * dx
        for j in range(J):
            y = j * dy
            # Bloc diélectrique haute permittivité
            if 0.1 <= x <= 0.5 and 0.45 <= y <= 0.55:
                eps_r[i, j] = 50.0

    return eps_r


# =============================================================================
# Classe Solveur
# =============================================================================

class Poisson2D:
    """
    Solveur éléments finis / différences finies pour l'équation de Poisson 2D.

    Résout : -∇·[ε_r ∇V] = ρ  sur Ω = [0,Lx]x[0,Ly]
    avec :   V = 0 sur ∂Ω  (Dirichlet homogène).

    La matrice du système A (de taille N²xN²) est construite une fois dans
    build_matrix() et réutilisée par tous les solveurs.

    Attributs :
        I, J     (int)         : Nombre de nœuds (x, y).
        Lx, Ly   (float)       : Dimensions physiques du domaine.
        dx, dy   (float)       : Pas de discrétisation.
        N        (int)         : Taille totale du système (IxJ).
        eps_r    (np.ndarray)  : Carte de permittivité (IxJ).
    """

    def __init__(
        self,
        I: int,
        J: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        eps_r_matrix: np.ndarray = None
    ) -> None:
        """
        Initialise le solveur avec les paramètres géométriques et physiques.

        IN:
            I            (int)          : Nombre de nœuds selon x.
            J            (int)          : Nombre de nœuds selon y.
            Lx           (float)        : Longueur physique selon x (défaut 1.0).
            Ly           (float)        : Longueur physique selon y (défaut 1.0).
            eps_r_matrix (np.ndarray)   : Carte de permittivité (IxJ).
        OUT:
            None
        """
        self.I, self.J   = I, J
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (I - 1)
        self.dy = Ly / (J - 1)
        self.N  = I * J
        # Si aucune carte fournie, on utilise le vide (ε_r = 1)
        self.eps_r = eps_r_matrix if eps_r_matrix is not None else np.ones((I, J))


    def get_k(self, i: int, j: int) -> int:
        """
        Convertit les indices 2D (i, j) en indice 1D k dans le vecteur d'état.

        IN:
            i (int) : Indice selon x ∈ [0, I-1].
            j (int) : Indice selon y ∈ [0, J-1].
        OUT:
            k (int) : Indice aplati ∈ [0, I*J-1].
        """
        return i * self.J + j


    def build_matrix(self, sparse: bool = True) -> sp.csr_matrix | np.ndarray:
        """
        Construit la matrice du système A de l'équation discrétisée.

        IN:
            sparse (bool) : Si True (défaut), retourne une matrice CSR.
        OUT:
            A (sp.csr_matrix | np.ndarray) : Matrice du système de taille N²xN².
        """
        A   = sp.dok_matrix((self.N, self.N), dtype=np.float64)
        idx = 1.0 / (self.dx**2)   # Facteur de pondération selon x
        idy = 1.0 / (self.dy**2)   # Facteur de pondération selon y

        for i in range(self.I):
            for j in range(self.J):
                k = self.get_k(i, j)

                if i == 0 or i == self.I - 1 or j == 0 or j == self.J - 1:
                    # Condition de Dirichlet : ligne identité
                    A[k, k] = 1.0
                else:
                    # Coefficients de diffusion locaux (moyenne arithmétique des ε_r voisins)
                    eE = (self.eps_r[i+1, j] + self.eps_r[i, j]) / 2.0 * idx
                    eW = (self.eps_r[i-1, j] + self.eps_r[i, j]) / 2.0 * idx
                    eN = (self.eps_r[i, j+1] + self.eps_r[i, j]) / 2.0 * idy
                    eS = (self.eps_r[i, j-1] + self.eps_r[i, j]) / 2.0 * idy

                    # Stencil à 5 points
                    A[k, k] = -(eE + eW + eN + eS)   # Diagonal (nœud central)
                    A[k, k + self.J] = eE            # Est  (i+1, j)
                    A[k, k - self.J] = eW            # Ouest (i-1, j)
                    A[k, k + 1] = eN                 # Nord  (i, j+1)
                    A[k, k - 1] = eS                 # Sud   (i, j-1)

        return A.tocsr() if sparse else A.toarray()


    def compute_residual(self, V_2d: np.ndarray, rho_vector: np.ndarray) -> float:
        """
        Calcule le résidu relatif de la solution : ||A V - b|| / ||b||.

        IN:
            V_2d       (np.ndarray) : Solution potentiel (IxJ), sera aplatie.
            rho_vector (np.ndarray) : Vecteur de charges source b = -ρ (taille N²).
        OUT:
            residual (float) : Résidu relatif. 
        """
        A = self.build_matrix(sparse=True)
        B = -rho_vector
        norm_B = np.linalg.norm(B)
        if norm_B == 0:
            return 0.0
        return np.linalg.norm(A.dot(V_2d.flatten()) - B) / norm_B


    def solve_direct_dense(self, rho_vector: np.ndarray) -> np.ndarray:
        """
        Résout le système A V = -ρ par factorisation LU dense (Pivot de Gauss).

        IN:
            rho_vector (np.ndarray) : Vecteur source ρ de taille N².
        OUT:
            V_2d (np.ndarray) : Potentiel solution mis en forme (IxJ).
        """
        print(f" -> Inversion Dense (Matrice {self.N}x{self.N})...")
        A_dense  = self.build_matrix(sparse=False)
        V_vector = np.linalg.solve(A_dense, -rho_vector)
        return V_vector.reshape((self.I, self.J))


    def solve_direct_sparse(self, rho_vector: np.ndarray) -> np.ndarray:
        """
        Résout le système A V = -ρ par factorisation directe creuse (SuperLU).

        IN:
            rho_vector (np.ndarray) : Vecteur source ρ de taille N².
        OUT:
            V_2d (np.ndarray) : Potentiel solution mis en forme (IxJ).
        """
        print(" -> Inversion Sparse Directe...")
        A_sparse = self.build_matrix(sparse=True)
        V_vector = spla.spsolve(A_sparse, -rho_vector)
        return V_vector.reshape((self.I, self.J))


    def solve_jacobi(self, rho_vector: np.ndarray, tol: float = 1e-5, max_iter: int = 5000,
        return_history: bool = False) -> np.ndarray | tuple:
        """
        Résout le système par la méthode itérative de Jacobi.

        IN:
            rho_vector     (np.ndarray) : Vecteur source ρ.
            tol            (float)      : Tolérance de convergence sur ||Δx||.
            max_iter       (int)        : Nombre maximal d'itérations.
            return_history (bool)       : Si True, retourne aussi l'historique du résidu.
        OUT:
            V_2d    (np.ndarray) : Potentiel solution (IxJ).
            history (list)       : [Si return_history=True] Résidus itération par itération.
        """
        A = self.build_matrix()
        b = -rho_vector
        D = A.diagonal()           # Diagonale principale
        R = A - sp.diags(D)        # Partie hors-diagonale
        x = np.zeros(self.N)       # Initialisation à zéro
        history = []

        for _ in range(max_iter):
            x_new = (b - R.dot(x)) / D   # Mise à jour de Jacobi (vectorisée)
            res = np.linalg.norm(x_new - x)
            if return_history:
                history.append(res)
            if res < tol:
                break
            x = x_new

        if return_history:
            return x.reshape((self.I, self.J)), history
        return x.reshape((self.I, self.J))


    def solve_sor(self, rho_vector: np.ndarray, omega: float = 1.0, tol: float = 1e-5,
        max_iter: int = 3000, return_history: bool = False) -> np.ndarray | tuple:
        """
        Résout le système par la méthode SOR (Sur-Relaxation Successive).

        IN:
            rho_vector     (np.ndarray) : Vecteur source ρ.
            omega          (float)      : Paramètre de relaxation (défaut 1.0 = Gauss-Seidel).
            tol            (float)      : Tolérance de convergence sur ||Δx||.
            max_iter       (int)        : Nombre maximal d'itérations.
            return_history (bool)       : Si True, retourne aussi l'historique du résidu.
        OUT:
            V_2d    (np.ndarray) : Potentiel solution (IxJ).
            history (list)       : [Si return_history=True] Résidus itération par itération.
        """
        A = self.build_matrix()
        b = -rho_vector
        D = sp.diags(A.diagonal())
        L = sp.tril(A, k=-1)    # Partie triangulaire inférieure stricte
        U = sp.triu(A, k=1)     # Partie triangulaire supérieure stricte

        # Pré-calcul des matrices de splitting
        M = (D + omega * L).tocsr()             # Matrice de splitting (triangulaire inférieure)
        N = (1 - omega) * D - omega * U         # Matrice de correction

        x       = np.zeros(self.N)
        history = []

        for _ in range(max_iter):
            rhs   = N.dot(x) + omega * b
            x_new = spla.spsolve_triangular(M, rhs, lower=True)   # Résolution triangulaire
            res   = np.linalg.norm(x_new - x)
            if return_history:
                history.append(res)
            if res < tol:
                break
            x = x_new

        if return_history:
            return x.reshape((self.I, self.J)), history
        return x.reshape((self.I, self.J))


    def solve_cg(self, rho_vector: np.ndarray, tol: float = 1e-5) -> np.ndarray:
        """
        Résout le système A V = -ρ par la méthode du Gradient Conjugué (CG).

        IN:
            rho_vector (np.ndarray) : Vecteur source ρ de taille N².
            tol        (float)      : Tolérance relative de convergence (défaut 1e-5).
        OUT:
            V_2d (np.ndarray) : Potentiel solution mis en forme (IxJ).
        """
        V_vector, _ = spla.cg(self.build_matrix(), -rho_vector, rtol=tol)
        return V_vector.reshape((self.I, self.J))
