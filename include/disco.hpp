/*
 * Disco C++ v0.1.4
 * https://github.com/ankane/disco-cpp
 * MIT License
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <version>

#if defined(__cpp_lib_ranges_chunk) || defined(__cpp_lib_ranges_enumerate) || defined(__cpp_lib_ranges_zip)
#include <ranges>
#endif

namespace disco {

namespace detail {

template<typename T>
class Map {
  private:
    std::unordered_map<T, size_t> map;
    std::vector<T> vec;

  public:
    size_t add(const T& id) {
        auto [it, inserted] = map.try_emplace(id, vec.size());
        if (inserted) {
            vec.push_back(id);
        }
        return it->second;
    }

    std::optional<size_t> get(const T& id) const {
        auto search = map.find(id);
        if (search != map.end()) {
            return search->second;
        }
        return std::nullopt;
    }

    const T& lookup(size_t index) const {
        return vec.at(index);
    }

    size_t size() const {
        return vec.size();
    }

    std::span<const T> ids() const {
        return std::span{vec};
    }
};

/// A dense matrix.
class DenseMatrix {
  private:
    std::vector<float> data;

  public:
    size_t rows;
    size_t cols;

    DenseMatrix(size_t rows, size_t cols) : rows{rows}, cols{cols} {
        data = std::vector<float>(rows * cols, 0);
    }

    std::span<const float> row(size_t i) const {
        // subspan does not perform bounds checking
        if (i >= rows) {
            throw std::out_of_range{"row out of range"};
        }
        size_t idx = i * cols;
        return std::span{data}.subspan(idx, cols);
    }

    std::span<float> row(size_t i) {
        // subspan does not perform bounds checking
        if (i >= rows) {
            throw std::out_of_range{"row out of range"};
        }
        size_t idx = i * cols;
        return std::span{data}.subspan(idx, cols);
    }

#if defined(__cpp_lib_ranges_chunk)
    auto rows_() const {
        return std::views::chunk(data, cols);
    }

    auto rows_() {
        return std::views::chunk(data, cols);
    }
#endif

    std::vector<float> dot(std::span<const float> x) const {
        std::vector<float> res;
        res.reserve(rows);
        // TODO allow compiler to auto-vectorize
#if defined(__cpp_lib_ranges_chunk) && defined(__cpp_lib_ranges_zip)
        for (const auto& r : rows_()) {
            float sum = 0.0;
            for (const auto& [rj, xj] : std::views::zip(r, x)) {
                sum += rj * xj;
            }
            res.push_back(sum);
        }
#else
        for (size_t i = 0; i < rows; i++) {
            std::span<const float> r = row(i);
            float sum = 0.0;
            for (size_t j = 0; j < cols; j++) {
                sum += r[j] * x[j];
            }
            res.push_back(sum);
        }
#endif
        return res;
    }
};

/// A coordinate list (COO) matrix.
class CooMatrix {
  private:
    // separate vectors to avoid padding
    std::vector<size_t> row_indices;
    std::vector<size_t> col_indices;
    std::vector<float> values;

  public:
    void reserve(size_t capacity) {
        row_indices.reserve(capacity);
        col_indices.reserve(capacity);
        values.reserve(capacity);
    }

    void push(size_t row_index, size_t col_index, float value) {
        row_indices.push_back(row_index);
        col_indices.push_back(col_index);
        values.push_back(value);
    }

    size_t size() const {
        return row_indices.size();
    }

    std::tuple<size_t, size_t, float> at(size_t i) {
        return {row_indices.at(i), col_indices.at(i), values.at(i)};
    }
};

/// A list of lists (LIL) matrix.
class LilMatrix {
  private:
    std::vector<std::vector<std::pair<size_t, float>>> row_list;

  public:
    void push(size_t row_index, size_t col_index, float value) {
        if (row_index == row_list.size()) {
            row_list.emplace_back();
        }
        row_list.at(row_index).emplace_back(col_index, value);
    }

    size_t size() const {
        return row_list.size();
    }

    std::vector<std::vector<std::pair<size_t, float>>>::iterator begin() {
        return row_list.begin();
    }

    std::vector<std::vector<std::pair<size_t, float>>>::iterator end() {
        return row_list.end();
    }
};

inline float norm(std::span<const float> a) {
    float sum = 0.0;
    // TODO allow compiler to auto-vectorize
    for (auto v : a) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

inline std::vector<float> norms(const DenseMatrix& factors) {
    std::vector<float> norms;
    norms.reserve(factors.rows);
#if defined(__cpp_lib_ranges_chunk)
    for (const auto& row : factors.rows_()) {
        norms.push_back(norm(row));
    }
#else
    for (size_t i = 0; i < factors.rows; i++) {
        norms.push_back(norm(factors.row(i)));
    }
#endif
    return norms;
}

inline float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0;
    // TODO allow compiler to auto-vectorize
#if defined(__cpp_lib_ranges_zip)
    for (const auto& [ai, bi] : std::views::zip(a, b)) {
        sum += ai * bi;
    }
#else
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
#endif
    return sum;
}

inline void scaled_add(std::span<float> x, float a, std::span<const float> v) {
#if defined(__cpp_lib_ranges_zip)
    for (auto&& [xi, vi] : std::views::zip(x, v)) {
        xi += a * vi;
    }
#else
    for (size_t i = 0; i < x.size(); i++) {
        x[i] += a * v[i];
    }
#endif
}

inline void neg(std::span<float> x) {
    for (auto& v : x) {
        v = -v;
    }
}

inline void least_squares_cg(LilMatrix& cui, DenseMatrix& x, DenseMatrix& y, float regularization) {
    size_t cg_steps = 3;

    // calculate YtY
    size_t factors = x.cols;
    DenseMatrix yty{factors, factors};
#if defined(__cpp_lib_ranges_chunk) && defined(__cpp_lib_ranges_enumerate) && defined(__cpp_lib_ranges_zip)
    for (auto&& [i, row] : std::views::enumerate(yty.rows_())) {
        for (auto&& [j, rowj] : std::views::enumerate(row)) {
            float sum = 0.0f;
            for (const auto& r : y.rows_()) {
                sum += r[i] * r[j];
            }
            rowj = sum;
        }
        row[i] += regularization;
    }
#else
    for (size_t i = 0; i < factors; i++) {
        std::span<float> row = yty.row(i);
        for (size_t j = 0; j < row.size(); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < y.rows; k++) {
                std::span<const float> r = y.row(k);
                sum += r[i] * r[j];
            }
            row[j] = sum;
        }
        row[i] += regularization;
    }
#endif

    size_t u = 0;
    for (auto& row_vec : cui) {
        // start from previous iteration
        std::span<float> xi = x.row(u);

        // calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
        std::vector<float> r = yty.dot(xi);
        neg(r);
        for (auto [i, confidence] : row_vec) {
            scaled_add(r, confidence - (confidence - 1.0f) * dot(y.row(i), xi), y.row(i));
        }

        std::vector<float> p(r);
        float rsold = dot(r, r);

        for (size_t step = 0; step < cg_steps; step++) {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            std::vector<float> ap = yty.dot(p);
            for (auto [i, confidence] : row_vec) {
                scaled_add(ap, (confidence - 1.0f) * dot(y.row(i), p), y.row(i));
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
#if defined(__cpp_lib_ranges_zip)
            for (auto&& [pi, ri] : std::views::zip(p, r)) {
                pi = ri + rs * pi;
            }
#else
            for (size_t i = 0; i < p.size(); i++) {
                p.at(i) = r.at(i) + rs * p.at(i);
            }
#endif
            rsold = rsnew;
        }
        u++;
    }
}

inline std::vector<size_t> sample(std::mt19937_64& prng, size_t n) {
    std::vector<size_t> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.push_back(i);
    }

    if (n > 0) {
        // Fisher–Yates shuffle
        std::uniform_real_distribution<float> dist(0, 1);
        for (size_t i = n - 1; i >= 1; i--) {
            auto j = static_cast<size_t>(dist(prng) * static_cast<float>(i + 1));
            std::swap(v.at(i), v.at(j));
        }
    }

    return v;
}

template<typename T>
std::vector<std::pair<T, float>> similar(
    const Map<T>& map,
    const DenseMatrix& factors,
    const std::vector<float>& norms,
    const T& id,
    size_t count
) {
    std::optional<size_t> io = map.get(id);
    if (!io) {
        return std::vector<std::pair<T, float>>();
    }

    size_t i = io.value();
    std::span<const float> query = factors.row(i);
    float query_norm = norms.at(i);

    std::vector<std::pair<size_t, float>> predictions;
    predictions.reserve(factors.rows);
#if defined(__cpp_lib_ranges_chunk) && defined(__cpp_lib_ranges_zip)
    for (const auto& [j, row, norm] : std::views::zip(std::views::iota(0), factors.rows_(), norms)) {
        predictions.emplace_back(j, dot(row, query) / (norm * query_norm));
    }
#else
    for (size_t j = 0; j < factors.rows; j++) {
        std::span<const float> row = factors.row(j);
        float score = dot(row, query) / (norms.at(j) * query_norm);
        predictions.emplace_back(j, score);
    }
#endif
    count = std::min(count, predictions.size() - 1);
    // TODO check cast
    auto diff = static_cast<ptrdiff_t>(std::min(predictions.size(), count + 1));
    std::ranges::partial_sort(
        predictions,
        predictions.begin() + diff,
        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second > b.second;
        }
    );

    std::vector<std::pair<T, float>> recs;
    recs.reserve(count);
    for (auto [index, score] : predictions) {
        if (index == i) {
            continue;
        }

        recs.emplace_back(map.lookup(index), score);

        if (recs.size() == count) {
            break;
        }
    }
    return recs;
}

} // namespace detail

/// A dataset.
template<typename T, typename U>
class Dataset {
  public:
    /// Creates a new dataset.
    Dataset() = default;

    /// Adds a rating to the dataset.
    void push(T user_id, U item_id, float value) {
        data.emplace_back(user_id, item_id, value);
    }

    /// Returns the number of ratings in the dataset.
    size_t size() const {
        return data.size();
    }

    /// @private
    std::vector<std::tuple<T, U, float>> data;
};

/// Information about a training iteration.
struct FitInfo {
    /// The iteration.
    size_t iteration;
    /// The training loss.
    float train_loss;
};

/// Recommender options.
struct RecommenderOptions {
    /// Sets the number of factors.
    size_t factors = 8;
    /// Sets the number of iterations.
    size_t iterations = 20;
    /// Sets the regularization.
    // there is regularization by default
    // but explicit and implicit have different defaults
    std::optional<float> regularization = std::nullopt;
    /// Sets the learning rate.
    float learning_rate = 0.1f;
    /// Sets alpha.
    float alpha = 40.0f;
    /// Sets the callback for each iteration.
    std::function<void(const FitInfo&)> callback = nullptr;
    /// Sets the random seed.
    std::optional<uint64_t> seed = std::nullopt;
};

/// A recommender.
template<typename T, typename U>
class Recommender {
  public:
    /// Creates a recommender with explicit feedback.
    static Recommender<T, U> fit_explicit(
        const Dataset<T, U>& train_set,
        const RecommenderOptions& options = RecommenderOptions()
    ) {
        return fit(train_set, options, false);
    }

    /// Creates a recommender with implicit feedback.
    static Recommender<T, U> fit_implicit(
        const Dataset<T, U>& train_set,
        const RecommenderOptions& options = RecommenderOptions()
    ) {
        return fit(train_set, options, true);
    }

    /// Returns the predicted rating for a specific user and item.
    float predict(const T& user_id, const U& item_id) const {
        std::optional<size_t> i = user_map_.get(user_id);
        if (!i) {
            return global_mean_;
        }

        std::optional<size_t> j = item_map_.get(item_id);
        if (!j) {
            return global_mean_;
        }

        return detail::dot(user_factors_.row(i.value()), item_factors_.row(j.value()));
    }

    /// Returns recommendations for a user.
    std::vector<std::pair<U, float>> user_recs(const T& user_id, size_t count = 5) const {
        std::optional<size_t> io = user_map_.get(user_id);
        if (!io) {
            return std::vector<std::pair<U, float>>();
        }

        size_t i = io.value();
        std::span<const float> query = user_factors_.row(i);

        const std::set<size_t>& rated = rated_.at(i);

        std::vector<std::pair<size_t, float>> predictions;
        predictions.reserve(item_factors_.rows);
#if defined(__cpp_lib_ranges_chunk) && defined(__cpp_lib_ranges_enumerate)
        for (const auto& [j, row] : std::views::enumerate(item_factors_.rows_())) {
            predictions.emplace_back(j, detail::dot(row, query));
        }
#else
        for (size_t j = 0; j < item_factors_.rows; j++) {
            float score = detail::dot(item_factors_.row(j), query);
            predictions.emplace_back(j, score);
        }
#endif
        count = std::min(count, predictions.size() - rated.size());
        // TODO check cast
        auto diff = static_cast<ptrdiff_t>(std::min(predictions.size(), count + rated.size()));
        std::ranges::partial_sort(
            predictions,
            predictions.begin() + diff,
            [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                return a.second > b.second;
            }
        );

        std::vector<std::pair<U, float>> recs;
        recs.reserve(count);
        for (auto [index, score] : predictions) {
            if (rated.contains(index)) {
                continue;
            }

            recs.emplace_back(item_map_.lookup(index), score);

            if (recs.size() == count) {
                break;
            }
        }
        return recs;
    }

    /// Returns recommendations for an item.
    std::vector<std::pair<U, float>> item_recs(const U& item_id, size_t count = 5) const {
        return similar<U>(item_map_, item_factors_, item_norms_, item_id, count);
    }

    /// Returns similar users.
    std::vector<std::pair<T, float>> similar_users(const T& user_id, size_t count = 5) const {
        return similar<T>(user_map_, user_factors_, user_norms_, user_id, count);
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
        std::optional<size_t> i = user_map_.get(user_id);
        if (i) {
            return user_factors_.row(i.value());
        }
        return std::nullopt;
    }

    /// Returns factors for a specific item.
    std::optional<std::span<const float>> item_factors(const U& item_id) const {
        std::optional<size_t> i = item_map_.get(item_id);
        if (i) {
            return item_factors_.row(i.value());
        }
        return std::nullopt;
    }

    /// Returns the global mean.
    float global_mean() const {
        return global_mean_;
    }

  private:
    detail::Map<T> user_map_;
    detail::Map<U> item_map_;
    std::vector<std::set<size_t>> rated_;
    float global_mean_;
    detail::DenseMatrix user_factors_;
    detail::DenseMatrix item_factors_;
    std::vector<float> user_norms_;
    std::vector<float> item_norms_;

    Recommender(
        detail::Map<T>&& user_map,
        detail::Map<U>&& item_map,
        std::vector<std::set<size_t>>&& rated,
        float global_mean,
        detail::DenseMatrix&& user_factors,
        detail::DenseMatrix&& item_factors,
        std::vector<float>&& user_norms,
        std::vector<float>&& item_norms
    ) :
        user_map_{std::move(user_map)},
        item_map_{std::move(item_map)},
        rated_{std::move(rated)},
        global_mean_{global_mean},
        user_factors_{std::move(user_factors)},
        item_factors_{std::move(item_factors)},
        user_norms_{std::move(user_norms)},
        item_norms_{std::move(item_norms)} {}

    static detail::DenseMatrix create_factors(
        size_t rows,
        size_t cols,
        std::mt19937_64& prng,
        float end_range
    ) {
        detail::DenseMatrix m{rows, cols};
        std::uniform_real_distribution<float> dist(0, end_range);
#if defined(__cpp_lib_ranges_chunk)
        for (auto&& row : m.rows_()) {
            for (auto& v : row) {
                v = dist(prng);
            }
        }
#else
        for (size_t i = 0; i < m.rows; i++) {
            for (auto& v : m.row(i)) {
                v = dist(prng);
            }
        }
#endif
        return m;
    }

    static Recommender<T, U> fit(
        const Dataset<T, U>& train_set,
        const RecommenderOptions& options,
        bool implicit
    ) {
        detail::Map<T> user_map;
        detail::Map<U> item_map;
        std::vector<std::set<size_t>> rated;

        detail::CooMatrix train_data;
        if (!implicit) {
            train_data.reserve(train_set.size());
        }
        float sum = 0.0f;

        detail::LilMatrix cui;
        detail::LilMatrix ciu;

        for (const auto& [user_id, item_id, value] : train_set.data) {
            size_t u = user_map.add(user_id);
            size_t i = item_map.add(item_id);

            if (implicit) {
                float confidence = 1.0f + options.alpha * value;
                cui.push(u, i, confidence);
                ciu.push(i, u, confidence);
            } else {
                train_data.push(u, i, value);
                sum += value;
            }

            if (u == rated.size()) {
                rated.emplace_back();
            }
            rated.at(u).insert(i);
        }

        float global_mean = implicit ? 0.0f : sum / static_cast<float>(train_data.size());

        size_t users = user_map.size();
        size_t items = item_map.size();
        size_t factors = options.factors;
        std::mt19937_64 prng;
        if (options.seed) {
            prng.seed(options.seed.value());
        }
        float end_range = implicit ? 0.01f : 0.1f;

        detail::DenseMatrix user_factors = create_factors(users, factors, prng, end_range);
        detail::DenseMatrix item_factors = create_factors(items, factors, prng, end_range);

        if (implicit) {
            // conjugate gradient method
            // https://www.benfrederickson.com/fast-implicit-matrix-factorization/

            float regularization = options.regularization.value_or(0.01f);

            for (size_t iteration = 0; iteration < options.iterations; iteration++) {
                least_squares_cg(cui, user_factors, item_factors, regularization);
                least_squares_cg(ciu, item_factors, user_factors, regularization);

                if (options.callback) {
                    FitInfo info{
                        .iteration = iteration + 1,
                        .train_loss = std::numeric_limits<float>::quiet_NaN()
                    };
                    options.callback(info);
                }
            }
        } else {
            // stochastic gradient method with twin learners
            // https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf
            // algorithm 2

            float learning_rate = options.learning_rate;
            float lambda = options.regularization.value_or(0.1f);
            size_t k = factors;
            size_t ks = std::max(
                static_cast<size_t>(std::round(static_cast<double>(k) * 0.08)),
                static_cast<size_t>(1)
            );

            std::vector<float> g_slow(users, 1.0f);
            std::vector<float> g_fast(users, 1.0f);
            std::vector<float> h_slow(items, 1.0f);
            std::vector<float> h_fast(items, 1.0f);

            for (size_t iteration = 0; iteration < options.iterations; iteration++) {
                float train_loss = 0.0f;

                // shuffle for each iteration
                for (auto j : detail::sample(prng, train_set.size())) {
                    auto [u, v, r] = train_data.at(j);

                    std::span<float> pu = user_factors.row(u);
                    std::span<float> qv = item_factors.row(v);
                    float e = r - detail::dot(pu, qv);

                    // slow learner
                    float g_hat = 0.0f;
                    float h_hat = 0.0f;

                    float nu = learning_rate * (1.0f / std::sqrt(g_slow.at(u)));
                    float nv = learning_rate * (1.0f / std::sqrt(h_slow.at(v)));

#if defined(__cpp_lib_ranges_zip)
                    for (auto&& [pud, qvd] : std::views::zip(pu.subspan(0, ks), qv.subspan(0, ks))) {
                        float gud = -e * qvd + lambda * pud;
                        float hvd = -e * pud + lambda * qvd;

                        g_hat += gud * gud;
                        h_hat += hvd * hvd;

                        pud -= nu * gud;
                        qvd -= nv * hvd;
                    }
#else
                    for (size_t d = 0; d < ks; d++) {
                        float gud = -e * qv[d] + lambda * pu[d];
                        float hvd = -e * pu[d] + lambda * qv[d];

                        g_hat += gud * gud;
                        h_hat += hvd * hvd;

                        pu[d] -= nu * gud;
                        qv[d] -= nv * hvd;
                    }
#endif

                    g_slow.at(u) += g_hat / static_cast<float>(ks);
                    h_slow.at(v) += h_hat / static_cast<float>(ks);

                    // fast learner
                    // don't update on first outer iteration
                    if (iteration > 0) {
                        float g_hat = 0.0f;
                        float h_hat = 0.0f;

                        float nu = learning_rate * (1.0f / std::sqrt(g_fast.at(u)));
                        float nv = learning_rate * (1.0f / std::sqrt(h_fast.at(v)));

#if defined(__cpp_lib_ranges_zip)
                        for (auto&& [pud, qvd] : std::views::zip(pu.subspan(ks), qv.subspan(ks))) {
                            float gud = -e * qvd + lambda * pud;
                            float hvd = -e * pud + lambda * qvd;

                            g_hat += gud * gud;
                            h_hat += hvd * hvd;

                            pud -= nu * gud;
                            qvd -= nv * hvd;
                        }
#else
                        for (size_t d = ks; d < k; d++) {
                            float gud = -e * qv[d] + lambda * pu[d];
                            float hvd = -e * pu[d] + lambda * qv[d];

                            g_hat += gud * gud;
                            h_hat += hvd * hvd;

                            pu[d] -= nu * gud;
                            qv[d] -= nv * hvd;
                        }
#endif

                        g_fast.at(u) += g_hat / static_cast<float>(k - ks);
                        h_fast.at(v) += h_hat / static_cast<float>(k - ks);
                    }

                    train_loss += e * e;
                }

                if (options.callback) {
                    train_loss = std::sqrt(train_loss / static_cast<float>(train_set.size()));

                    FitInfo info{.iteration = iteration + 1, .train_loss = train_loss};
                    options.callback(info);
                }
            }
        }

        std::vector<float> user_norms = detail::norms(user_factors);
        std::vector<float> item_norms = detail::norms(item_factors);

        return Recommender<T, U>(
            std::move(user_map),
            std::move(item_map),
            std::move(rated),
            global_mean,
            std::move(user_factors),
            std::move(item_factors),
            std::move(user_norms),
            std::move(item_norms)
        );
    }
};

} // namespace disco
