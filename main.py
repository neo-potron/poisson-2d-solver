"""
main.py — Point d'entrée du projet Poisson 2D.

Ce module présente un menu interactif permettant à l'utilisateur de choisir
parmi plusieurs méthodes de résolution de l'équation de Poisson 2D :
    - Méthodes directes (Dense et Sparse)
    - Méthodes itératives (Jacobi, Gauss-Seidel, SOR avec optimisation de ω)
    - Gradient Conjugué
    - Études comparatives (performance et précision en fonction de N)
    - Analyse de convergence et d'erreur
    - Simulation avec milieu diélectrique (permittivité variable)

Auteurs : Néo Potron & Alex Cassi
Date    : 24/04/2026
Cours   : Méthodes Numériques
"""

import time
from poisson import Poisson2D, setup_charges, setup_permittivity
from analysis import (
    plot_potential,
    plot_electric_field,
    plot_electric_field_norm,
    find_optimal_omega,
    run_performance_study,
    run_precision_study,
    plot_convergence_comparison,
    plot_error_analysis,
    plot_charges
)


def main() -> None:
    """
    Fonction principale : affiche le menu interactif, instancie le solveur
    et dispatch vers la routine de résolution choisie par l'utilisateur.

    Le maillage par défaut est une grille de 65x65 nœuds sur un domaine
    unitaire [0,1]x[0,1]. Les charges sont initialisées par setup_charges().

    IN:
        None
    OUT:
        None
    """

    N, Lx, Ly = 65, 1.0, 1.0                  # Taille de grille et dimensions physiques
    rho_vec = setup_charges(N, N, Lx, Ly)     # Vecteur de charges (aplati, N²)
    solver = Poisson2D(N, N, Lx, Ly)          # Solveur par défaut (ε_r = 1 partout)

    print("\n" + "="*55)
    print("  PROJET POISSON 2D - MÉTHODES NUMÉRIQUES")
    print("="*55)
    print("1.  Inversion DIRECTE (Dense - Pivot Gauss)")
    print("2.  Inversion DIRECTE (Sparse)")
    print("3.  JACOBI")
    print("4.  GAUSS-SEIDEL")
    print("5.  SUR-RELAXATION (SOR) + OPTIMISATION w")
    print("6.  GRADIENT CONJUGUÉ")
    print("-------------------------------------------------------")
    print("7.  Étude de temps : Comparaison globale")
    print("8.  Étude de précision : Comparaison globale")
    print("9.  Analyse de convergence et erreur")
    print("10. Simulation avec un diélectrique")

    choix = input("\nEntrez votre choix (1-10) : ")

    V_2d = None      # Grille 2D du potentiel (IxJ), remplie selon la méthode choisie
    eps_map = None   # Carte de permittivité

    start_global = time.time()   # Chronomètre global lancé avant la résolution

    if choix == '1':
        # Résolution par inversion dense (Pivot de Gauss via np.linalg.solve).
        V_2d = solver.solve_direct_dense(rho_vec)

    elif choix == '2':
        # Résolution par factorisation sparse directe (SuperLU via scipy.sparse.linalg.spsolve).
        V_2d = solver.solve_direct_sparse(rho_vec)

    elif choix == '3':
        # Méthode de Jacobi
        V_2d = solver.solve_jacobi(rho_vec)

    elif choix == '4':
        # Gauss-Seidel (= SOR avec ω = 1.0)
        V_2d = solver.solve_sor(rho_vec, omega=1.0)

    elif choix == '5':
        # Sur-Relaxation Optimale 
        w_opt = find_optimal_omega(solver, rho_vec)
        start_global = time.time()
        V_2d = solver.solve_sor(rho_vec, omega=w_opt)

    elif choix == '6':
        # Gradient Conjugué : méthode de Krylov
        V_2d = solver.solve_cg(rho_vec)

    elif choix == '7':
        # Étude de performance : mesure du temps de calcul pour N ∈ [16, 256]
        run_performance_study()
        return

    elif choix == '8':
        # Étude de précision : calcul du résidu relatif pour N ∈ [16, 256]
        run_precision_study()
        return

    elif choix == '9':
        # Analyse approfondie :
        # Comparaison de la vitesse de convergence (Jacobi / GS / SOR)
        # Carte spatiale de l'erreur absolue par rapport à la solution directe
        print("\n -> Analyse profonde en cours...")
        V_dir = solver.solve_direct_sparse(rho_vec)
        _, h_jac = solver.solve_jacobi(rho_vec, return_history=True)
        V_gs, h_gs = solver.solve_sor(rho_vec, omega=1.0,  return_history=True)
        w_opt = 1.86   # Valeur théorique de l'optimum pour N=65
        V_sor, h_sor = solver.solve_sor(rho_vec, omega=w_opt, return_history=True)
        plot_convergence_comparison(h_jac, h_gs, h_sor)
        plot_error_analysis(V_dir, V_sor, Lx, Ly)
        return
    
    elif choix == '10':
        # Résolution dans un milieu hétérogène : le domaine contient un bloc diélectrique (ε_r = 50)
        print("\n -> Création du milieu diélectrique...")
        eps_map = setup_permittivity(N, N, Lx, Ly)
        solver_dielec = Poisson2D(N, N, Lx, Ly, eps_r_matrix=eps_map)
        V_2d = solver_dielec.solve_cg(rho_vec)
        solver = solver_dielec   # Remplacement du solveur pour le calcul du résidu

    else:
        print("Choix invalide.")
        return

    if V_2d is not None:
        temps_total = time.time() - start_global
        residu = solver.compute_residual(V_2d, rho_vec)

        print(f"\n================ BILAN ==================")
        print(f"Temps de calcul total : {temps_total:.4f} secondes")
        print(f"Précision (Résidu)    : {residu:.2e}")
        print(f"=========================================")

        print("Génération de la carte des charges...")
        plot_charges(rho_vec, N, N, Lx, Ly)

        print("Génération des cartes de potentiel et de champ...")
        plot_potential(V_2d, Lx, Ly)

        suffix = "_dielec" if choix == '10' else ""
        plot_electric_field(V_2d, eps_r_matrix=eps_map, Lx=Lx, Ly=Ly, suffix=suffix)
        plot_electric_field_norm(V_2d, eps_r_matrix=eps_map, Lx=Lx, Ly=Ly, suffix=suffix)

if __name__ == "__main__":
    main()
