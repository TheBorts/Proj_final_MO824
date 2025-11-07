#include "KMedoidsEvaluator.h"

KMedoidsEvaluator::KMedoidsEvaluator(const vector<vector<double>>& D, int k)
    : D_(D), n_(static_cast<int>(D.size())), k_(k)
{
}

double KMedoidsEvaluator::avg_from_medoids(const vector<int>& medoids) const
{
    if (medoids.empty())
    {
        return numeric_limits<double>::infinity();
    }
    double total = 0.0;
    for (int i = 0; i < n_; ++i)
    {
        double best = numeric_limits<double>::infinity();
        for (int m : medoids)
        {
            best = min(best, D_[i][m]);
        }
        total += best;
    }
    return total / static_cast<double>(n_);
}

double KMedoidsEvaluator::base_avg(const Solution<int>& sol) const
{
    if (sol.empty())
    {
        return numeric_limits<double>::infinity();
    }

    if (isfinite(sol.cost))
    {
        return sol.cost;
    }
    vector<int> meds(sol.begin(), sol.end());

    return avg_from_medoids(meds);
}

double KMedoidsEvaluator::evaluate(const Solution<int>& sol) const
{
    vector<int> meds(sol.begin(), sol.end());

    return avg_from_medoids(meds);
}

double KMedoidsEvaluator::evaluate_insertion_cost(const int& elem, const Solution<int>& sol) const
{
    if (contains(sol, elem))
    {
        return numeric_limits<double>::infinity();
    }
    vector<int> meds(sol.begin(), sol.end());
    meds.push_back(elem);

    double new_avg = avg_from_medoids(meds);
    double base = base_avg(sol);

    return isinf(base) ? new_avg : (new_avg - base);
}

double KMedoidsEvaluator::evaluate_removal_cost(const int& elem, const Solution<int>& sol) const
{
    if (!contains(sol, elem) || sol.size() <= 1)
    {
        return numeric_limits<double>::infinity();
    }
    vector<int> meds;
    meds.reserve(sol.size() - 1);
    for (int v : sol)
    {
        if (v != elem) meds.push_back(v);
    }

    double new_avg = avg_from_medoids(meds);
    double base = base_avg(sol);

    return new_avg - base;
}

double KMedoidsEvaluator::evaluate_exchange_cost(const int& elem_in, const int& elem_out,
                                                 const Solution<int>& sol) const
{
    if (!contains(sol, elem_out) || contains(sol, elem_in))
    {
        return numeric_limits<double>::infinity();
    }
    vector<int> meds(sol.begin(), sol.end());
    auto it = find(meds.begin(), meds.end(), elem_out);
    *it = elem_in;

    double new_avg = avg_from_medoids(meds);
    double base = base_avg(sol);

    return new_avg - base;
}
