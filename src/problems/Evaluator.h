#pragma once
#include "../solutions/Solution.h"

template <typename T>
class Evaluator
{
   public:
    virtual ~Evaluator() = default;

    virtual int get_domain_size() const = 0;
    virtual double evaluate(const Solution<T>& sol) const = 0;
    virtual double evaluate_insertion_cost(const T& elem, const Solution<T>& sol) const = 0;
    virtual double evaluate_removal_cost(const T& elem, const Solution<T>& sol) const = 0;
    virtual double evaluate_exchange_cost(const T& elem_in, const T& elem_out,
                                          const Solution<T>& sol) const = 0;
};
