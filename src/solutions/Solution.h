#pragma once
#include <initializer_list>
#include <limits>
#include <ostream>
#include <sstream>
#include <vector>

using namespace std;

template <typename T>
class Solution : public vector<T>
{
   public:
    using Base = vector<T>;
    double cost;

    Solution() : Base(), cost(numeric_limits<double>::infinity()) {}
    Solution(const Solution& other) : Base(other), cost(other.cost) {}

    template <typename It>
    Solution(It first, It last) : Base(first, last), cost(numeric_limits<double>::infinity())
    {
    }

    Solution(initializer_list<T> ilist) : Base(ilist), cost(numeric_limits<double>::infinity()) {}

    void add(const T& elem) { this->push_back(elem); }
    size_t size() const { return Base::size(); }
    Solution copy() const { return Solution(*this); }

    string str() const
    {
        ostringstream oss;
        oss << "Solution: cost=[" << cost << "], size=[" << this->size() << "], elements=[";
        for (size_t i = 0; i < this->size(); ++i)
        {
            if (i) oss << ", ";
            oss << (*this)[i];
        }
        oss << "]";
        return oss.str();
    }
};

template <typename T>
inline ostream& operator<<(ostream& os, const Solution<T>& s)
{
    return os << s.str();
}
