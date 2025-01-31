#include <algorithm>
#include <cassert>
#include <span>
#include <string>

#include "../include/disco.hpp"

using disco::Dataset;
using disco::Recommender;

template<typename T> void assert_eq(const std::vector<T>& a, const std::vector<T>& b) {
    assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));
}

template<typename T> void assert_eq(std::span<const T> a, const std::vector<T>& b) {
    assert(std::equal(a.begin(), a.end(), b.begin(), b.end()));
}

void test_rated() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(1, "C", 1.0);
    data.push(1, "D", 1.0);
    data.push(2, "C", 1.0);
    data.push(2, "D", 1.0);
    data.push(2, "E", 1.0);
    data.push(2, "F", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);

    std::vector<std::string> item_ids;
    for (auto& v : recommender.user_recs(1, 5)) {
        item_ids.push_back(v.first);
    }
    std::sort(item_ids.begin(), item_ids.end());
    assert_eq(item_ids, {"E", "F"});

    item_ids.clear();
    for (auto& v : recommender.user_recs(2, 5)) {
        item_ids.push_back(v.first);
    }
    std::sort(item_ids.begin(), item_ids.end());
    assert_eq(item_ids, {"A", "B"});
}

void test_item_recs_same_score() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "C", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);
    std::vector<std::string> item_ids;
    for (auto& v : recommender.item_recs("A", 5)) {
        item_ids.push_back(v.first);
    }
    assert_eq(item_ids, {"B", "C"});
}

void test_ids() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "B", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data);
    assert_eq(recommender.user_ids(), {1, 2});
    assert_eq(recommender.item_ids(), {"A", "B"});
}

void test_factors() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);
    data.push(1, "B", 1.0);
    data.push(2, "B", 1.0);

    auto recommender = Recommender<int, std::string>::fit_implicit(data, { .factors = 20 });

    assert((*recommender.user_factors(1)).size() == 20);
    assert((*recommender.item_factors("A")).size() == 20);

    assert(recommender.user_factors(3) == std::nullopt);
    assert(recommender.item_factors("C") == std::nullopt);
}

void test_callback_explicit() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);

    size_t calls = 0;
    auto callback = [&calls](const disco::FitInfo& info) {
        assert(info.iteration == calls + 1);
        assert(!std::isnan(info.train_loss));
        calls++;
    };
    Recommender<int, std::string>::fit_explicit(data, { .callback = callback });
    assert(calls == 20);
}

void test_callback_implicit() {
    auto data = Dataset<int, std::string>();
    data.push(1, "A", 1.0);

    size_t calls = 0;
    auto callback = [&calls](const disco::FitInfo& info) {
        assert(info.iteration == calls + 1);
        assert(std::isnan(info.train_loss));
        calls++;
    };
    Recommender<int, std::string>::fit_implicit(data, { .callback = callback });
    assert(calls == 20);
}

int main() {
    test_rated();
    test_item_recs_same_score();
    test_ids();
    test_factors();
    test_callback_explicit();
    test_callback_implicit();
    return 0;
}
