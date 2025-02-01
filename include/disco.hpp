/*!
 * Disco C++ v0.1.3
 * https://github.com/ankane/disco-cpp
 * MIT License
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <unordered_map>
#include <vector>

namespace disco {

namespace {

template<typename T, typename U> struct Rating
{
    T user_id;
    U item_id;
    float value;
};

template<typename T> class Map
{
public:
    size_t add(const T& id) {
        auto search = map_.find(id);
        if (search != map_.end()) {
            return search->second;
        } else {
            size_t i = vec_.size();
            map_.emplace(std::make_pair(id, i));
            vec_.push_back(id);
            return i;
        }
    }

    std::optional<size_t> get(const T& id) const {
        auto search = map_.find(id);
        if (search != map_.end()) {
            return std::optional<size_t>{search->second};
        }
        return std::nullopt;
    }

    const T& lookup(size_t index) const {
        return vec_.at(index);
    }

    size_t size() const {
        return vec_.size();
    }

    std::span<const T> ids() const {
        return std::span{vec_};
    }

    std::unordered_map<T, size_t> map_;
    std::vector<T> vec_;
};

class Matrix
{
public:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;

    Matrix(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_ = std::vector<float>(rows * cols, 0);
    }

    std::span<const float> row(size_t i) const {
        size_t idx = i * cols_;
        return std::span{data_}.subspan(idx, cols_);
    }

    std::span<float> row_mut(size_t i) {
        size_t idx = i * cols_;
        return std::span{data_}.subspan(idx, cols_);
    }

    std::vector<float> dot(std::span<float> x) {
        std::vector<float> res;
        res.reserve(rows_);
        for (size_t i = 0; i < rows_; i++) {
            auto r = row(i);
            float sum = 0.0;
            for (size_t j = 0; j < cols_; j++) {
                sum += r[j] * x[j];
            }
            res.push_back(sum);
        }
        return res;
    }
};

inline float norm(std::span<const float> a) {
    float sum = 0.0;
    // TODO allow compiler to auto-vectorize
    for (auto& v : a) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

inline float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0;
    // TODO allow compiler to auto-vectorize
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void scaled_add(std::span<float> x, float a, std::span<const float> v) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] += a * v[i];
    }
}

inline void neg(std::span<float> x) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = -x[i];
    }
}

inline void least_squares_cg(std::vector<std::vector<std::pair<size_t, float>>>& cui, Matrix& x, Matrix& y, float regularization) {
    size_t cg_steps = 3;

    // calculate YtY
    size_t factors = x.cols_;
    Matrix yty = Matrix(factors, factors);
    for (size_t i = 0; i < factors; i++) {
        for (size_t j = 0; j < factors; j++) {
            float sum = 0.0;
            for (size_t k = 0; k < y.rows_; k++) {
                sum += y.data_[k * factors + i] * y.data_[k * factors + j];
            }
            yty.data_[i * factors + j] = sum;
        }
    }
    for (size_t i = 0; i < factors; i++) {
        yty.data_[i * factors + i] += regularization;
    }

    for (size_t u = 0; u < cui.size(); u++) {
        auto row_vec = cui[u];

        // start from previous iteration
        auto xi = x.row_mut(u);

        // calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
        auto r = yty.dot(xi);
        neg(r);
        for (auto [i, confidence] : row_vec) {
            scaled_add(
                r,
                confidence - (confidence - 1.0) * dot(y.row(i), xi),
                y.row(i)
            );
        }

        std::vector<float> p(r);
        float rsold = dot(r, r);

        for (size_t step = 0; step < cg_steps; step++) {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            std::span sp(p);
            auto ap = yty.dot(sp);
            for (auto [i, confidence] : row_vec) {
                scaled_add(ap, (confidence - 1.0) * dot(y.row(i), p), y.row(i));
            }

            // standard CG update
            float alpha = rsold / dot(p, ap);
            scaled_add(xi, alpha, p);
            scaled_add(r, -alpha, ap);
            float rsnew = dot(r, r);

            if (rsnew < 1e-20) {
                break;
            }

            float rs = rsnew / rsold;
            for (size_t i = 0; i < p.size(); i++) {
                p[i] = r[i] + rs * p[i];
            }
            rsold = rsnew;
        }
    }
}

inline std::vector<size_t> sample(std::mt19937_64& prng, size_t n) {
    std::vector<size_t> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.push_back(i);
    }

    // Fisher–Yates shuffle
    std::uniform_real_distribution<float> dist(0, 1);
    for (size_t i = n - 1; i >= 1; i--) {
        size_t j = dist(prng) * (i + 1);
        std::swap(v[i], v[j]);
    }

    return v;
}

template<typename T> void truncate(std::vector<T>& vec, size_t count) {
    if (vec.size() > count) {
        vec.resize(count);
    }
}

template<typename T> std::vector<std::pair<T, float>> similar(const Map<T>& map, const Matrix& factors, const std::vector<float>& norms, const T& id, size_t count) {
    auto io = map.get(id);
    if (!io) {
        return std::vector<std::pair<T, float>>();
    }

    size_t i = *io;
    auto query = factors.row(i);
    auto query_norm = norms.at(i);

    std::vector<std::pair<size_t, float>> predictions;
    predictions.reserve(factors.rows_);
    for (size_t j = 0; j < factors.rows_; j++) {
        auto row = factors.row(j);
        float score = dot(row, query) / (norms.at(j) * query_norm);
        predictions.emplace_back(std::make_pair(j, score));
    }
    std::sort(predictions.begin(), predictions.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
        return a.second > b.second;
    });
    truncate(predictions, count + 1);
    std::erase_if(predictions, [&i](const std::pair<size_t, float>& v) { return v.first == i; });
    truncate(predictions, count);

    std::vector<std::pair<T, float>> recs;
    recs.reserve(predictions.size());
    for (auto& prediction : predictions) {
        recs.emplace_back(std::make_pair(map.lookup(prediction.first), prediction.second));
    }
    return recs;
}

}

/// A dataset.
template<typename T, typename U> class Dataset
{
public:
    /// Creates a new dataset.
    Dataset() {}

    /// Adds a rating to the dataset.
    void push(T user_id, U item_id, float value) {
        data_.emplace_back(Rating<T, U>{user_id, item_id, value});
    }

    /// Returns the number of ratings in the dataset.
    size_t size() const {
        return data_.size();
    }

    /// @private
    std::vector<Rating<T, U>> data_;
};

/// Information about a training iteration.
struct FitInfo
{
    /// The iteration.
    size_t iteration;
    /// The training loss.
    float train_loss;
};

/// Recommender options.
struct RecommenderOptions
{
    /// Sets the number of factors.
    size_t factors = 8;
    /// Sets the number of iterations.
    size_t iterations = 20;
    /// Sets the regularization.
    // there is regularization by default
    // but explicit and implicit have different defaults
    std::optional<float> regularization = std::nullopt;
    /// Sets the learning rate.
    float learning_rate = 0.1;
    /// Sets alpha.
    float alpha = 40.0;
    /// Sets the callback for each iteration.
    std::function<void (const FitInfo&)> callback = nullptr;
    /// Sets the random seed.
    std::optional<uint64_t> seed = std::nullopt;
};

/// A recommender.
template<typename T, typename U> class Recommender
{
public:
    /// Creates a recommender with explicit feedback.
    static Recommender<T, U> fit_explicit(const Dataset<T, U>& train_set, const RecommenderOptions& options = RecommenderOptions()) {
        return fit(train_set, options, false);
    }

    /// Creates a recommender with implicit feedback.
    static Recommender<T, U> fit_implicit(const Dataset<T, U>& train_set, const RecommenderOptions& options = RecommenderOptions()) {
        return fit(train_set, options, true);
    }

    /// Returns the predicted rating for a specific user and item.
    float predict(const T& user_id, const U& item_id) const {
        auto i = user_map_.get(user_id);
        if (!i) {
            return global_mean_;
        }

        auto j = item_map_.get(item_id);
        if (!j) {
            return global_mean_;
        }

        return dot(user_factors_.row(*i), item_factors_.row(*j));
    }

    /// Returns recommendations for a user.
    std::vector<std::pair<U, float>> user_recs(const T& user_id, size_t count = 5) const {
        auto io = user_map_.get(user_id);
        if (!io) {
            return std::vector<std::pair<U, float>>();
        }

        size_t i = *io;
        auto query = user_factors_.row(i);

        auto rated = rated_.find(i)->second;

        std::vector<std::pair<size_t, float>> predictions;
        predictions.reserve(item_factors_.rows_);
        for (size_t j = 0; j < item_factors_.rows_; j++) {
            float score = dot(item_factors_.row(j), query);
            predictions.emplace_back(std::make_pair(j, score));
        }
        std::sort(predictions.begin(), predictions.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second > b.second;
        });
        truncate(predictions, count + rated.size());
        std::erase_if(predictions, [&](const std::pair<size_t, float>& v) { return rated.contains(v.first); });
        truncate(predictions, count);

        std::vector<std::pair<U, float>> recs;
        recs.reserve(predictions.size());
        for (auto& prediction : predictions) {
            recs.emplace_back(std::make_pair(item_map_.lookup(prediction.first), prediction.second));
        }
        return recs;
    }

    /// Returns recommendations for an item.
    std::vector<std::pair<U, float>> item_recs(const U& item_id, size_t count = 5) const {
        return similar<U>(
            item_map_,
            item_factors_,
            item_norms_,
            item_id,
            count
        );
    }

    /// Returns similar users.
    std::vector<std::pair<T, float>> similar_users(const T& user_id, size_t count = 5) const {
        return similar<T>(
            user_map_,
            user_factors_,
            user_norms_,
            user_id,
            count
        );
    }

    /// Returns user ids.
    std::span<const T> user_ids() const {
        return user_map_.ids();
    }

    /// Returns item ids.
    std::span<const U> item_ids() const {
        return item_map_.ids();
    }

    /// Returns factors for a specific user.
    std::optional<std::span<const float>> user_factors(const T& user_id) const {
        auto i = user_map_.get(user_id);
        if (i) {
            return user_factors_.row(*i);
        }
        return std::nullopt;
    }

    /// Returns factors for a specific item.
    std::optional<std::span<const float>> item_factors(const U& item_id) const {
        auto i = item_map_.get(item_id);
        if (i) {
            return item_factors_.row(*i);
        }
        return std::nullopt;
    }

    /// Returns the global mean.
    float global_mean() const {
        return global_mean_;
    }

private:
    Map<T> user_map_;
    Map<U> item_map_;
    std::unordered_map<size_t, std::set<size_t>> rated_;
    float global_mean_;
    Matrix user_factors_;
    Matrix item_factors_;
    std::vector<float> user_norms_;
    std::vector<float> item_norms_;

    Recommender(Map<T> user_map, Map<U> item_map, std::unordered_map<size_t, std::set<size_t>> rated, float global_mean, Matrix user_factors, Matrix item_factors)
        : user_map_{user_map}, item_map_{item_map}, rated_{rated}, global_mean_{global_mean}, user_factors_{user_factors}, item_factors_{item_factors} {}

    static Matrix create_factors(size_t rows, size_t cols, std::mt19937_64& prng, float end_range) {
        auto m = Matrix(rows, cols);
        std::uniform_real_distribution<float> dist(0, end_range);
        for (size_t i = 0; i < m.data_.size(); i++) {
            m.data_[i] = dist(prng);
        }
        return m;
    }

    static Recommender<T, U> fit(const Dataset<T, U>& train_set, const RecommenderOptions& options, bool implicit) {
        size_t factors = options.factors;

        Map<T> user_map;
        Map<U> item_map;
        std::unordered_map<size_t, std::set<size_t>> rated;

        std::vector<size_t> row_inds;
        std::vector<size_t> col_inds;
        std::vector<float> values;

        row_inds.reserve(train_set.size());
        col_inds.reserve(train_set.size());
        values.reserve(train_set.size());

        std::vector<std::vector<std::pair<size_t, float>>> cui;
        std::vector<std::vector<std::pair<size_t, float>>> ciu;

        for (auto& rating : train_set.data_) {
            size_t u = user_map.add(rating.user_id);
            size_t i = item_map.add(rating.item_id);

            if (implicit) {
                if (u == cui.size()) {
                    cui.emplace_back(std::vector<std::pair<size_t, float>>());
                }

                if (i == ciu.size()) {
                    ciu.emplace_back(std::vector<std::pair<size_t, float>>());
                }

                float confidence = 1.0 + options.alpha * rating.value;
                cui[u].emplace_back(std::make_pair(i, confidence));
                ciu[i].emplace_back(std::make_pair(u, confidence));
            } else {
                row_inds.push_back(u);
                col_inds.push_back(i);
                values.push_back(rating.value);
            }

            auto search = rated.find(u);
            if (search == rated.end()) {
                rated.insert(std::make_pair(u, std::set<size_t>{i}));
            } else {
                search->second.insert(i);
            }
        }

        size_t users = user_map.size();
        size_t items = item_map.size();

        float global_mean = implicit ? 0.0 : (std::reduce(values.begin(), values.end()) / values.size());

        float end_range = implicit ? 0.01 : 0.1;

        std::mt19937_64 prng;
        if (options.seed) {
            prng.seed(*options.seed);
        }

        Matrix user_factors = create_factors(users, factors, prng, end_range);
        Matrix item_factors = create_factors(items, factors, prng, end_range);

        auto recommender = Recommender<T, U>(
            user_map,
            item_map,
            rated,
            global_mean,
            user_factors,
            item_factors
        );

        if (implicit) {
            // conjugate gradient method
            // https://www.benfrederickson.com/fast-implicit-matrix-factorization/

            float regularization = options.regularization.value_or(0.01);

            for (size_t iteration = 0; iteration < options.iterations; iteration++) {
                least_squares_cg(
                    cui,
                    recommender.user_factors_,
                    recommender.item_factors_,
                    regularization
                );
                least_squares_cg(
                    ciu,
                    recommender.item_factors_,
                    recommender.user_factors_,
                    regularization
                );

                if (options.callback) {
                    FitInfo info;
                    info.iteration = iteration + 1;
                    info.train_loss = std::numeric_limits<float>::quiet_NaN();
                    options.callback(info);
                }
            }
        } else {
            // stochastic gradient method with twin learners
            // https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf
            // algorithm 2

            float learning_rate = options.learning_rate;
            float lambda = options.regularization.value_or(0.1);
            size_t k = factors;
            size_t ks = std::max((size_t) std::round(k * 0.08), (size_t) 1);

            std::vector<float> g_slow(users, 1.0);
            std::vector<float> g_fast(users, 1.0);
            std::vector<float> h_slow(items, 1.0);
            std::vector<float> h_fast(items, 1.0);

            for (size_t iteration = 0; iteration < options.iterations; iteration++) {
                float train_loss = 0.0;

                // shuffle for each iteration
                for (auto& j : sample(prng, train_set.size())) {
                    size_t u = row_inds[j];
                    size_t v = col_inds[j];

                    auto pu = recommender.user_factors_.row_mut(u);
                    auto qv = recommender.item_factors_.row_mut(v);
                    float e = values[j] - dot(pu, qv);

                    // slow learner
                    float g_hat = 0.0;
                    float h_hat = 0.0;

                    float nu = learning_rate * (1.0 / std::sqrt(g_slow[u]));
                    float nv = learning_rate * (1.0 / std::sqrt(h_slow[v]));

                    for (size_t d = 0; d < ks; d++) {
                        float gud = -e * qv[d] + lambda * pu[d];
                        float hvd = -e * pu[d] + lambda * qv[d];

                        g_hat += gud * gud;
                        h_hat += hvd * hvd;

                        pu[d] -= nu * gud;
                        qv[d] -= nv * hvd;
                    }

                    g_slow[u] += g_hat / ks;
                    h_slow[v] += h_hat / ks;

                    // fast learner
                    // don't update on first outer iteration
                    if (iteration > 0) {
                        float g_hat = 0.0;
                        float h_hat = 0.0;

                        float nu = learning_rate * (1.0 / std::sqrt(g_fast[u]));
                        float nv = learning_rate * (1.0 / std::sqrt(h_fast[v]));

                        for (size_t d = ks; d < k; d++) {
                            float gud = -e * qv[d] + lambda * pu[d];
                            float hvd = -e * pu[d] + lambda * qv[d];

                            g_hat += gud * gud;
                            h_hat += hvd * hvd;

                            pu[d] -= nu * gud;
                            qv[d] -= nv * hvd;
                        }

                        g_fast[u] += g_hat / (k - ks);
                        h_fast[v] += h_hat / (k - ks);
                    }

                    train_loss += e * e;
                }

                if (options.callback) {
                    train_loss = std::sqrt(train_loss / train_set.size());

                    FitInfo info;
                    info.iteration = iteration + 1;
                    info.train_loss = train_loss;
                    options.callback(info);
                }
            }
        }

        for (size_t i = 0; i < users; i++) {
            recommender.user_norms_.push_back(norm(recommender.user_factors_.row(i)));
        }

        for (size_t i = 0; i < items; i++) {
            recommender.item_norms_.push_back(norm(recommender.item_factors_.row(i)));
        }

        return recommender;
    }
};

}
