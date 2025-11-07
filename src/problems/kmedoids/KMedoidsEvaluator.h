#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "../../problems/Evaluator.h"
#include "../../solutions/Solution.h"

using namespace std;

class KMedoidsEvaluator : public Evaluator<int>
{
   public:
    KMedoidsEvaluator(const vector<vector<double>>& D, int k);

    int get_domain_size() const override { return n_; }

    double evaluate(const Solution<int>& sol) const override;

    double evaluate_insertion_cost(const int& elem, const Solution<int>& sol) const override;

    double evaluate_removal_cost(const int& elem, const Solution<int>& sol) const override;

    double evaluate_exchange_cost(const int& elem_in, const int& elem_out,
                                  const Solution<int>& sol) const override;

   private:
    vector<vector<double>> D_;
    int n_{0};
    int k_{0};

    static bool contains(const Solution<int>& sol, int x)
    {
        return find(sol.begin(), sol.end(), x) != sol.end();
    }

    double avg_from_medoids(const vector<int>& medoids) const;
    double base_avg(const Solution<int>& sol) const;
};
