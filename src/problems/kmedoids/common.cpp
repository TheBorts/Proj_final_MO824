#include "problems/kmedoids/common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

inline void ltrim(string& s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char ch) { return !isspace(ch); }));
}
inline void rtrim(string& s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !isspace(ch); }).base(),
            s.end());
}
inline void trim(string& s)
{
    ltrim(s);
    rtrim(s);
}

double to_float(string token, char decimal)
{
    trim(token);
    if (decimal != '.')
    {
        replace(token.begin(), token.end(), decimal, '.');
    }

    return stod(token);
}

vector<vector<double>> load_i_dataset(const string& path, char sep, char decimal)
{
    ifstream fin(path);
    vector<vector<double>> rows;

    if (!fin.is_open())
    {
        return rows;
    }

    string line;
    while (getline(fin, line))
    {
        trim(line);
        if (line.empty() || (!line.empty() && line[0] == '#'))
        {
            continue;
        }

        vector<string> parts;
        {
            stringstream ss(line);
            string tok;
            while (getline(ss, tok, sep))
            {
                parts.push_back(tok);
            }
        }

        vector<double> row;
        row.reserve(parts.size());
        bool ok = true;
        try
        {
            for (auto& p : parts)
            {
                row.push_back(to_float(p, decimal));
            }
        }
        catch (...)
        {
            ok = false;
        }
        if (ok) rows.push_back(move(row));
    }

    if (rows.empty()) return rows;

    size_t d = rows.front().size();

    vector<vector<double>> filtered;

    filtered.reserve(rows.size());
    for (auto& r : rows)
    {
        if (r.size() == d) filtered.push_back(move(r));
    }
    return filtered;
}

vector<vector<double>>& zscore_inplace(vector<vector<double>>& X, int ddof)
{
    size_t n = X.size();

    if (n == 0) return X;

    size_t d = X[0].size();

    vector<double> means(d, 0.0);
    for (auto& row : X)
    {
        for (size_t j = 0; j < d; ++j)
        {
            means[j] += row[j];
        }
    }
    for (size_t j = 0; j < d; ++j)
    {
        means[j] /= static_cast<double>(n);
    }

    vector<double> sds(d, 1.0);
    if (n > 1)
    {
        vector<double> acc(d, 0.0);
        for (auto& row : X)
        {
            for (size_t j = 0; j < d; ++j)
            {
                double diff = row[j] - means[j];
                acc[j] += diff * diff;
            }
        }
        double denom = max<double>(1.0, static_cast<double>(n - ddof));
        for (size_t j = 0; j < d; ++j)
        {
            sds[j] = (acc[j] > 0.0) ? sqrt(acc[j] / denom) : 1.0;
        }
    }

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            double sd = (sds[j] != 0.0) ? sds[j] : 1.0;
            X[i][j] = (X[i][j] - means[j]) / sd;
        }
    }
    return X;
}

vector<vector<double>> pairwise_euclidean(const vector<vector<double>>& X)
{
    size_t n = X.size();
    vector<vector<double>> D(n, vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i)
    {
        auto& xi = X[i];

        for (size_t j = i + 1; j < n; ++j)
        {
            auto& xj = X[j];
            double s = 0.0;
            size_t d = min(xi.size(), xj.size());
            for (size_t k = 0; k < d; ++k)
            {
                double diff = xi[k] - xj[k];
                s += diff * diff;
            }
            double dist = sqrt(s);
            D[i][j] = dist;
            D[j][i] = dist;
        }
    }
    return D;
}
