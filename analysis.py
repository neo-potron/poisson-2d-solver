"""
analysis.py — Visualisation, optimisation et benchmarking du solveur Poisson 2D.

Ce module regroupe toutes les fonctions d'analyse post-traitement :

    Visualisation :
        - plot_charges()               : Carte de la distribution de charge ρ(x,y)
        - plot_potential()             : Heatmap du potentiel électrostatique V(x,y)
        - plot_electric_field()        : Lignes de champ E avec contours V
        - plot_electric_field_norm()   : Norme |E| avec lignes de champ

    Optimisation :
        - find_optimal_omega()         : Recherche du paramètre SOR optimal par balayage

    Benchmarking :
        - run_performance_study()      : Temps de calcul en fonction de N (jusqu'à 256)
        - run_precision_study()        : Résidu relatif en fonction de N

    Analyse avancée :
        - plot_convergence_comparison(): Vitesse de convergence des méthodes itératives
        - plot_error_analysis()        : Carte spatiale d'erreur et corrélation Directe/Itérative

Auteurs : Néo Potron & Alex Cassi
Date    : 24/04/2026
Cours   : Méthodes Numériques
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from poisson import Poisson2D, setup_charges

plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 12,
    'axes.labelsize' : 14,
    'axes.titlesize' : 15,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})


# =============================================================================
# 1. VISUALISATION  (Heatmaps & Champs)
# =============================================================================

def plot_charges(
    rho_vector: np.ndarray,
    I: int,
    J: int,
    Lx: float = 1.0,
    Ly: float = 1.0
) -> None:
    """
    Représente la distribution de charge ρ(x,y) sous forme de heatmap.

    IN:
        rho_vector (np.ndarray) : Vecteur aplati de charges (taille IxJ).
        I          (int)        : Nombre de nœuds selon x.
        J          (int)        : Nombre de nœuds selon y.
        Lx         (float)      : Extension physique selon x (défaut 1.0).
        Ly         (float)      : Extension physique selon y (défaut 1.0).
    OUT:
        None
    """
    # Reconstruction de la grille 2D depuis le vecteur aplati
    rho_2d   = rho_vector.reshape((I, J))
    x, y     = np.linspace(0, Lx, I), np.linspace(0, Ly, J)
    X, Y     = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, rho_2d, levels=50, cmap='bwr')
    plt.colorbar(contour, label='Densité de charge $\\rho$ (u.a.)')
    plt.title('Distribution des charges $\\rho(x,y)$')
    plt.xlabel('x (u.a.)')
    plt.ylabel('y (u.a.)')
    plt.tight_layout()
    plt.show()


def plot_potential(V_2d: np.ndarray, Lx: float = 1.0, Ly: float = 1.0) -> None:
    """
    Représente le potentiel électrostatique V(x,y) sous forme de heatmap à contours.

    IN:
        V_2d (np.ndarray) : Potentiel solution sur la grille (IxJ).
        Lx   (float)      : Extension physique selon x (défaut 1.0).
        Ly   (float)      : Extension physique selon y (défaut 1.0).
    OUT:
        None 
    """
    I, J = V_2d.shape
    x, y = np.linspace(0, Lx, I), np.linspace(0, Ly, J)
    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, V_2d, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, label='Potentiel V (u.a.)')
    plt.title('Distribution du potentiel électrostatique')
    plt.xlabel('x (u.a.)')
    plt.ylabel('y (u.a.)')
    plt.tight_layout()
    plt.show()


def plot_electric_field(
    V_2d: np.ndarray,
    eps_r_matrix: np.ndarray = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    suffix: str = ""
) -> None:
    """
    Représente les lignes de champ électrique E = -∇V superposées au potentiel.

    IN:
        V_2d         (np.ndarray) : Potentiel solution (IxJ).
        eps_r_matrix (np.ndarray) : Carte de permittivité (IxJ).
        Lx           (float)      : Extension physique selon x (défaut 1.0).
        Ly           (float)      : Extension physique selon y (défaut 1.0).
        suffix       (str)        : Suffixe du nom de fichier (ex: '_dielec').
    OUT:
        None 
    """
    I, J   = V_2d.shape
    x, y   = np.linspace(0, Lx, I), np.linspace(0, Ly, J)
    X, Y   = np.meshgrid(x, y, indexing='ij')
    # Calcul du champ E = -grad(V) via différences finies d'ordre 2
    Ex, Ey = np.gradient(-V_2d, Lx / (I - 1), Ly / (J - 1))

    plt.figure(figsize=(9, 7))
    plt.contourf(X, Y, V_2d, levels=50, cmap='RdBu_r', alpha=0.5)
    plt.colorbar(label='Potentiel V (u.a.)')

    # Tracé de la frontière diélectrique (si applicable)
    if eps_r_matrix is not None and np.max(eps_r_matrix) > 1.0:
        plt.contour(X, Y, eps_r_matrix, levels=[1.5],
                    colors='lime', linewidths=2, linestyles='--')

    plt.streamplot(X.T, Y.T, Ex.T, Ey.T, color='k', linewidth=1.0, density=1.5)
    plt.title('Lignes de champ électrique')
    plt.xlabel('x (u.a.)')
    plt.ylabel('y (u.a.)')
    plt.tight_layout()
    plt.show()


def plot_electric_field_norm(
    V_2d: np.ndarray,
    eps_r_matrix: np.ndarray = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    suffix: str = ""
) -> None:
    """
    Représente la norme du champ électrique |E|(x,y) avec un plafonnement à 95%.

    IN:
        V_2d         (np.ndarray) : Potentiel solution (IxJ).
        eps_r_matrix (np.ndarray) : Carte de permittivité (IxJ).
        Lx           (float)      : Extension physique selon x (défaut 1.0).
        Ly           (float)      : Extension physique selon y (défaut 1.0).
        suffix       (str)        : Suffixe du nom de fichier.
    OUT:
        None 
    """
    I, J = V_2d.shape
    x, y = np.linspace(0, Lx, I), np.linspace(0, Ly, J)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Ex, Ey = np.gradient(-V_2d, Lx / (I - 1), Ly / (J - 1))
    E_norm = np.sqrt(Ex**2 + Ey**2)
    # Plafonnement à 95% pour atténuer les artefacts aux singularités de charge
    E_max = np.percentile(E_norm, 95)

    plt.figure(figsize=(9, 7))
    plt.contourf(X, Y, E_norm, levels=50, cmap='magma', vmax=E_max)
    plt.colorbar(label='Norme du champ électrique |E|')

    # Frontière diélectrique en blanc
    if eps_r_matrix is not None and np.max(eps_r_matrix) > 1.0:
        plt.contour(X, Y, eps_r_matrix, levels=[1.5],
                    colors='lime', linewidths=2, linestyles='--')

    plt.streamplot(X.T, Y.T, Ex.T, Ey.T, color='white', linewidth=0.8, density=1.2)
    plt.title('Norme du champ électrique')
    plt.xlabel('x (u.a.)')
    plt.ylabel('y (u.a.)')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. OPTIMISATION ET BENCHMARKS
# =============================================================================

def find_optimal_omega(
    solver: Poisson2D,
    rho_vector: np.ndarray,
    w_min: float = 1.2,
    w_max: float = 1.95,
    num_points: int = 100
) -> float:
    """
    Détermine le paramètre de sur-relaxation optimal ω* par balayage temporel.

    IN:
        solver     (Poisson2D) : Instance du solveur configuré.
        rho_vector (np.ndarray): Vecteur source ρ.
        w_min      (float)     : Borne inférieure de ω (défaut 1.2).
        w_max      (float)     : Borne supérieure de ω (défaut 1.95).
        num_points (int)       : Nombre de points de balayage (défaut 20).
    OUT:
        best_w (float) : Valeur de ω minimisant le temps de résolution.
    """
    omegas = np.linspace(w_min, w_max, num_points)
    times = []
    print(f"\nBalayage de {num_points} valeurs de w...")

    for w in omegas:
        start = time.time()
        _ = solver.solve_sor(rho_vector, omega=w)
        times.append(time.time() - start)

    best_w = omegas[np.argmin(times)]

    plt.figure(figsize=(8, 5))
    plt.plot(omegas, times, 'o-', color='#2ca02c', linewidth=2)
    plt.axvline(best_w, color='red', linestyle='--',
                label=rf'Optimum : $\omega$ = {best_w:.2f}')
    plt.title('Performance de la Sur-Relaxation')
    plt.xlabel(r'Paramètre $\omega$')
    plt.ylabel('Temps (s)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

    return best_w


def run_performance_study() -> None:
    """
    Mesure et compare le temps de calcul des six méthodes en fonction de N.

    IN:
        None 
    OUT:
        None 
    """
    N_list = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256]

    methods = [
        'Direct (Dense)',
        'Direct (Sparse)',
        'Jacobi',
        'Gauss-Seidel',
        'SOR (w=1.86)',
        'Gradient Conjugué'
    ]
    results = {m: [] for m in methods}

    print(f"\nLancement du benchmark (N jusqu'à {max(N_list)})")

    for N in N_list:
        print(f"Calcul pour N = {N} (Grille {N}x{N})")
        solver = Poisson2D(N, N)
        rho_vec = setup_charges(N, N)

        # Direct DENSE (limité à N ≤ 160 pour la RAM)
        if N <= 160:
            try:
                start = time.time()
                _ = solver.solve_direct_dense(rho_vec)
                results['Direct (Dense)'].append(time.time() - start)
            except MemoryError:
                print(f"Mémoire saturée pour Dense à N={N}")
                results['Direct (Dense)'].append(np.nan)
        else:
            results['Direct (Dense)'].append(np.nan)  # Marqueur d'absence de donnée

        # Direct SPARSE
        start = time.time()
        _ = solver.solve_direct_sparse(rho_vec)
        results['Direct (Sparse)'].append(time.time() - start)

        # JACOBI
        start = time.time()
        _ = solver.solve_jacobi(rho_vec)
        results['Jacobi'].append(time.time() - start)

        # GAUSS-SEIDEL (SOR avec ω = 1)
        start = time.time()
        _ = solver.solve_sor(rho_vec, omega=1.0)
        results['Gauss-Seidel'].append(time.time() - start)

        # SOR (ω = 1.86)
        start = time.time()
        _ = solver.solve_sor(rho_vec, omega=1.86)
        results['SOR (w=1.86)'].append(time.time() - start)

        # GRADIENT CONJUGUÉ
        start = time.time()
        _ = solver.solve_cg(rho_vec)
        results['Gradient Conjugué'].append(time.time() - start)

    # Graphique Log-Log
    plt.figure(figsize=(12, 8))
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#000000']
    markers = ['o', 's', '^', 'D', 'v', '*']

    for i, method in enumerate(methods):
        mask = ~np.isnan(results[method])   # Filtre les NaN (méthode Dense tronquée)
        plt.plot(
            np.array(N_list)[mask],
            np.array(results[method])[mask],
            marker=markers[i], color=colors[i], label=method,
            linewidth=2.5, markersize=7, alpha=0.8
        )

    plt.yscale('log')
    plt.xscale('log')   # Double échelle log pour lire les pentes de complexité
    plt.title("Benchmarking de l'équation de Poisson", fontsize=14)
    plt.xlabel('Taille du maillage N (Log)', fontsize=12)
    plt.ylabel('Temps de calcul en secondes (Log)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.show()


def run_precision_study() -> None:
    """
    Mesure et compare la précision (résidu relatif) des six méthodes en fonction de N.

    IN:
        None
    OUT:
        None
    """
    N_list = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256]

    methods = [
        'Direct (Dense)',
        'Direct (Sparse)',
        'Jacobi',
        'Gauss-Seidel',
        'SOR (w=1.86)',
        'Gradient Conjugué'
    ]
    results = {m: [] for m in methods}

    print(f"\nLancement du benchmark (N jusqu'à {max(N_list)})")

    for N in N_list:
        print(f"--- Calcul de précision pour N = {N} ---")
        solver = Poisson2D(N, N)
        rho_vec = setup_charges(N, N)

        # Direct DENSE (sécurité RAM)
        if N <= 160:
            try:
                V = solver.solve_direct_dense(rho_vec)
                results['Direct (Dense)'].append(solver.compute_residual(V, rho_vec))
            except MemoryError:
                print(f"Mémoire saturée pour Dense à N={N}")
                results['Direct (Dense)'].append(np.nan)
        else:
            results['Direct (Dense)'].append(np.nan)

        # Autres méthodes 
        results['Direct (Sparse)'].append(
            solver.compute_residual(solver.solve_direct_sparse(rho_vec), rho_vec))
        results['Jacobi'].append(
            solver.compute_residual(solver.solve_jacobi(rho_vec), rho_vec))
        results['Gauss-Seidel'].append(
            solver.compute_residual(solver.solve_sor(rho_vec, omega=1.0), rho_vec))
        results['SOR (w=1.86)'].append(
            solver.compute_residual(solver.solve_sor(rho_vec, omega=1.8), rho_vec))
        results['Gradient Conjugué'].append(
            solver.compute_residual(solver.solve_cg(rho_vec), rho_vec))

    # Graphique Log-Log 
    plt.figure(figsize=(12, 8))
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#000000']
    markers = ['o', 's', '^', 'D', 'v', '*']

    for i, method in enumerate(methods):
        mask = ~np.isnan(results[method])
        # Plancher à 1e-16 pour éviter log(0) si la méthode directe est parfaite
        valid_y = np.maximum(np.array(results[method])[mask], 1e-16)

        plt.plot(
            np.array(N_list)[mask], valid_y,
            marker=markers[i], color=colors[i], label=method,
            linewidth=2.5, markersize=7, alpha=0.8
        )

    plt.yscale('log')
    plt.xscale('log')
    # Ligne de tolérance de référence
    plt.axhline(y=1e-5, color='gray', linestyle=':',
                label='Tolérance cible ($10^{-5}$)', linewidth=2)
    plt.title("Benchmarking de la précision numérique", fontsize=14)
    plt.xlabel('Taille du maillage N (Log)', fontsize=12)
    plt.ylabel('Erreur résiduelle relative (Log)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. ANALYSE DE CONVERGENCE ET D'ERREUR
# =============================================================================

def plot_convergence_comparison(
    hist_jacobi: list,
    hist_gs: list,
    hist_sor: list
) -> None:
    """
    Compare la vitesse de convergence des trois méthodes itératives.

    IN:
        hist_jacobi (list) : Historique des résidus de Jacobi par itération.
        hist_gs     (list) : Historique des résidus de Gauss-Seidel.
        hist_sor    (list) : Historique des résidus de SOR (ω optimal).
    OUT:
        None 
    """
    plt.figure(figsize=(10, 6))
    plt.plot(hist_jacobi, label='Jacobi', color='orange')
    plt.plot(hist_gs, label=r'Gauss-Seidel ($\omega=1$)', color='purple')
    plt.plot(hist_sor, label='SOR', color='green', linewidth=2)
    plt.yscale('log')
    plt.title('Vitesse de convergence des méthodes itératives')
    plt.xlabel('Itérations')
    plt.ylabel('Résidu (Erreur)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_analysis(
    V_ref: np.ndarray,
    V_comp: np.ndarray,
    Lx: float = 1.0,
    Ly: float = 1.0
) -> None:
    """
    Analyse spatiale de l'erreur absolue entre méthode directe et itérative.

    IN:
        V_ref  (np.ndarray) : Potentiel de référence (solution directe sparse, IxJ).
        V_comp (np.ndarray) : Potentiel à comparer (solution itérative, IxJ).
        Lx     (float)      : Extension physique selon x (défaut 1.0).
        Ly     (float)      : Extension physique selon y (défaut 1.0).
    OUT:
        None
    """
    error_map = np.abs(V_ref - V_comp)
    I, J = V_ref.shape
    x, y = np.linspace(0, Lx, I), np.linspace(0, Ly, J)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gauche : carte spatiale d'erreur
    im = ax1.contourf(X, Y, error_map, levels=50, cmap='inferno')
    plt.colorbar(im, ax=ax1, label='Erreur $|V_{dir} - V_{iter}|$')
    ax1.set_title("Carte Spatiale de l'Erreur Résiduelle")
    ax1.set_xlabel('x (u.a.)')
    ax1.set_ylabel('y (u.a.)')

    # Droite : corrélation Directe vs Itérative
    ax2.scatter(V_ref.flatten(), V_comp.flatten(), alpha=0.1, s=1, color='blue')
    lims = [np.min(V_ref), np.max(V_ref)]
    ax2.plot(lims, lims, 'r--', label='Ligne Idéale ($y=x$)')
    ax2.set_title('Corrélation : Méthode Directe vs Itérative')
    ax2.set_xlabel('Potentiel Direct (référence)')
    ax2.set_ylabel('Potentiel Itératif')
    ax2.legend()

    plt.tight_layout()
    plt.show()
