#include "GRASP_KMedoids_RW.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

GRASP_KMedoids_RW::GRASP_KMedoids_RW(double alpha, int iterations, const vector<vector<double>>& D,
                                     int k, LSSearch mode)
    : GRASP_KMedoids(alpha, iterations, D, k),
      D_(D),
      n_((int) D.size()),
      m_(D.empty() ? 0 : (int) D[0].size()),
      k_(k),
      mode_(mode)
{
    phi1.assign(n_, -1);
    d1.assign(n_, numeric_limits<double>::infinity());
    d2.assign(n_, numeric_limits<double>::infinity());
    gain.assign(m_, 0.0);
    loss.assign(m_, 0.0);
}

bool GRASP_KMedoids_RW::in_solution(int f, const vector<int>& S)
{
    return find(S.begin(), S.end(), f) != S.end();
}

void GRASP_KMedoids_RW::buildAssignments(const vector<int>& S)
{
    fill(d1.begin(), d1.end(), numeric_limits<double>::infinity());
    fill(d2.begin(), d2.end(), numeric_limits<double>::infinity());
    fill(phi1.begin(), phi1.end(), -1);

    servedBy.assign(m_, {});

    for (int u = 0; u < n_; ++u)
    {
        for (int idx = 0; idx < (int) S.size(); ++idx)
        {
            int f = S[idx];
            double du = D_[u][f];
            if (du < d1[u])
            {
                d2[u] = d1[u];
                d1[u] = du;
                phi1[u] = f;
            }
            else if (du < d2[u])
            {
                d2[u] = du;
            }
        }

        if (phi1[u] != -1) servedBy[phi1[u]].push_back(u);
    }
}

void GRASP_KMedoids_RW::buildGainLoss(const vector<int>& S)
{
    fill(gain.begin(), gain.end(), 0.0);
    fill(loss.begin(), loss.end(), 0.0);

    for (int fr : S)
    {
        double sum = 0.0;
        for (int u : servedBy[fr])
        {
            sum += (d2[u] - d1[u]);
        }
        loss[fr] = sum;
    }

    for (int fi = 0; fi < m_; ++fi)
    {
        if (in_solution(fi, S)) continue;

        double sum = 0.0;
        for (int u = 0; u < n_; ++u)
        {
            double improve = d1[u] - D_[u][fi];
            if (improve > 0.0) sum += improve;
        }
        gain[fi] = sum;
    }
}

double GRASP_KMedoids_RW::extra_pair(int fi, int fr, const vector<int>& S)
{
    double sum = 0.0;
    auto& users = servedBy[fr];
    for (int u : users)
    {
        double dufi = D_[u][fi];
        if (dufi < d2[u])
        {
            double term = d2[u] - max(dufi, d1[u]);
            if (term > 0) sum += term;
        }
    }
    return sum;
}

void GRASP_KMedoids_RW::apply_swap_update(int fi, int fr, vector<int>& S)
{
    auto it = find(S.begin(), S.end(), fr);
    if (it != S.end()) 
        *it = fi;

    for (int u : servedBy[fr])
    {
        double best = numeric_limits<double>::infinity();
        double second = numeric_limits<double>::infinity();
        int bestf = -1;
        for (int f : S)
        {
            double du = D_[u][f];
            if (du < best)
            {
                second = best;
                best = du;
                bestf = f;
            }
            else if (du < second)
            {
                second = du;
            }
        }
        d1[u] = best;
        d2[u] = second;
        phi1[u] = bestf;
    }

    for (int u = 0; u < n_; ++u)
    {
        double du = D_[u][fi];
        if (du < d1[u])
        {
            d2[u] = d1[u];
            d1[u] = du;
            phi1[u] = fi;
        }
        else if (du < d2[u] && phi1[u] != fi)
        {
            d2[u] = du;
        }
    }

    servedBy[fr].clear();
    servedBy[fi].clear();
    for (int u = 0; u < n_; ++u)
    {
        if (phi1[u] == fi)
            servedBy[fi].push_back(u);
        else if (phi1[u] == fr)
            servedBy[fr].push_back(u);
    }

    buildGainLoss(S);
}

Solution<int> GRASP_KMedoids_RW::localSearch()
{
    auto& S = *sol;
    if ((int) S.size() != k_) return *sol;

    buildAssignments(S);
    buildGainLoss(S);

    const double eps = 1e-12;
    bool improved = true;

    while (improved)
    {
        improved = false;

        double bestDelta = 0.0;
        int best_in = -1, best_out = -1;

        vector<int> out_list = S;

        vector<int> in_list;
        in_list.reserve(m_ - (int) S.size());
        for (int f = 0; f < m_; ++f)
            if (!in_solution(f, S)) in_list.push_back(f);

        if (mode_ == LSSearch::FirstImproving)
        {
            for (int fi : in_list)
            {
                for (int fr : out_list)
                {
                    double prof = gain[fi] - loss[fr] + extra_pair(fi, fr, S);
                    if (prof > eps)
                    {
                        apply_swap_update(fi, fr, S);
                        sol->cost = ObjFunction.evaluate(S);
                        improved = true;
                        break;
                    }
                }
                if (improved) break;
            }
        }
        else
        {
            for (int fi : in_list)
            {
                for (int fr : out_list)
                {
                    double prof = gain[fi] - loss[fr] + extra_pair(fi, fr, S);
                    double delta = -prof;
                    if (delta < bestDelta - eps)
                    {
                        bestDelta = delta;
                        best_in = fi;
                        best_out = fr;
                    }
                }
            }
            if (best_in != -1)
            {
                apply_swap_update(best_in, best_out, S);
                sol->cost = ObjFunction.evaluate(S);
                improved = true;
            }
        }
    }

    return *sol;
}
