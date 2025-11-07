#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../solutions/Solution.h"
#include "../Evaluator.h"

using namespace std;

class KMedoids : public Evaluator<int>
{
   public:
        KMedoids(const vector<vector<double>>& D, int k);
        int get_domain_size() const override { return n_; }
        double evaluate(const Solution<int>& sol) const override;
        double evaluate_insertion_cost(const int& p, const Solution<int>& sol) const override;
        double evaluate_removal_cost(const int& q, const Solution<int>& sol) const override;
        double evaluate_exchange_cost(const int& p, const int& q, const Solution<int>& sol) const override;

   private:
        vector<vector<double>> D_;
        int n_{0};
        int k_{0};

        mutable vector<int> medoids_signature_;
        mutable vector<double> nearest_dist_;
        mutable vector<int> nearest_medoid_;
        mutable vector<double> second_dist_;

        void sync_state(Solution<int>& sol);

        bool contains(Solution<int>& sol, int x){ return find(sol.begin(), sol.end(), x) != sol.end();}

        vector<int> sorted_signature(Solution<int>& sol)
        {
            vector<int> sig(sol.begin(), sol.end());
            sort(sig.begin(), sig.end());
            return sig;
        }
};
