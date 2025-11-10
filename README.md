Projeto final da matéria de Pesquisa Operacional - MO824.

Autores:

Eliel Lucas de Oliveira Carvalho - e295745@dac.unicamp.br

Lucas Ribeiro Bortoletto         - l173422@dac.unicamp.br



compilar:

GRASP:
g++ -std=c++17 -I src  src/problems/kmedoids/solvers/Main_Experiments.cpp  src/problems/kmedoids/solvers/GRASP_KMedoids.cpp  src/problems/kmedoids/solvers/GRASP_KMedoids_FI.cpp  src/problems/kmedoids/solvers/GRASP_KMedoids_POP.cpp  src/problems/kmedoids/solvers/GRASP_KMedoids_RPG.cpp  src/problems/kmedoids/solvers/GRASP_KMedoids_RW.cpp  src/problems/kmedoids/KMedoidsEvaluator.cpp  src/problems/kmedoids/common.cpp  -o run_grasp


Gurobi:

python modelo_pli.py