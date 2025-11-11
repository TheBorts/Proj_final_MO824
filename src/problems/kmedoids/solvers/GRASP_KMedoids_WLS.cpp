#include "GRASP_KMedoids_WLS.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

GRASP_KMedoids_WLS::GRASP_KMedoids_WLS(double alpha, int iterations, const vector<vector<double>>& D,
                                     int k, LSSearch mode)
    : GRASP_KMedoids(alpha, iterations, D, k),
      D_(D),
      n_((int) D.size()),
      m_(D.empty() ? 0 : (int) D[0].size()),
      k_(k),
      mode_(mode)
{
    vector<int> assignments;
    vector<vector<tuple<int,double>>> summed_distances;
}

void GRASP_KMedoids_WLS::buildAssignments(const vector<int>& S)
{
    assignments.resize(n_);
    for (int u = 0; u < n_; ++u)
    {
        for (int i = 0; i < k_; ++i)
        {
            int f = S[i];
            if (i == 0 || D_[u][f] < D_[u][assignments[u]])
            {
                assignments[u] = f;
            }
        }
    }
}

void GRASP_KMedoids_WLS::buildSummedDistances(const vector<int>& S)
{
    summed_distances.clear();
    summed_distances.resize(m_);

    for (int u = 0; u < n_; ++u)
    {
        int medoid = assignments[u];
        for (int i = 0; i < n_; ++i)
        {
            if (i != medoid) continue;
            summed_distances[i] += D_[u][i];
        }
    }
}


void GRASP_KMedoids_WLS::updateSummedDistances_swappoint(int p_out, int where_to, const vector<int>& S)
{
    int old_medoid = assignments[p_out];
    assignments[p_out] = where_to;

    summed_distances[p_out] = 0;

    for (int u = 0; u < n_; ++u)
    {
        if (assignments[u] == old_medoid)
        {
            summed_distances[u] -= D_[u][p_out];
        }
        if (assignments[u] == where_to)
        {
            summed_distances[u] += D_[u][p_out];
            summed_distances[p_out] += D_[u][p_out];
        }
    }
}


void GRASP_KMedoids_WLS::iterateConvergence(vector<int>& S)
{
    bool changed = false;

    for (int u = 0; u < n_; u++){
        int medoid = assignments[u];
        for(int i = 0; i < k_; i++){
            int f = S[i];
            if (D_[u][f] < D_[u][medoid]){
                medoid = f;
                changed = true;
            }
        }
        updateSummedDistances_swappoint(u, medoid, S);
        assignments[u] = medoid;
    }

    if (!changed) return;

    changed = false;
    vector<double> minimal_distance;
    vector<int> best_indices;
    minimal_distance.assign(k_, numeric_limits<double>::infinity());
    best_indices.assign(k_, -1);
    for (int u = 0; u < n_; ++u)
    {
        if (minimal_distance[assignments[u]] > summed_distances[u])
        {
            minimal_distance[assignments[u]] = summed_distances[u];
            best_indices[assignments[u]] = u;
        }
    }
    for (int i = 0; i < k_; ++i)
    {
        if (best_indices[i] != -1 && S[i] != best_indices[i])
        {
            S[i] = best_indices[i];
            changed = true;
        }
    }

    if (changed)
        iterateConvergence(S);
}


Solution<int> GRASP_KMedoids_WLS::localSearch()
{
    auto& S = *sol;
    if ((int) S.size() != k_) return *sol;

    buildAssignments(S);
    buildSummedDistances(S);

    iterateConvergence(S);

    return *sol;
}
