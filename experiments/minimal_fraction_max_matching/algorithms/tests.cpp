// tests to the cpp version to see that it works correctly
#include <catch2/catch_all.hpp>
#include "minimal_fraction_max_matching.cpp"   // bring the algorithm in-line
// compile : ❯ g++ -std=c++17 -O2 tests.cpp -lCatch2Main -lCatch2 -o tests
#include <boost/graph/random.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <random>

using Catch::Approx;
using Graph  = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

// helper: sum of incident weights at vertex v
static double incident_sum(std::uint32_t v,
                           const FractionalMatchingSolver::Matching& m)
{
    double s = 0.0;
    for (const auto& [e, val] : m)
        if (e.u == v || e.v == v) s += val;
    return s;
}

// ------------------------------ 1  Empty graph ------------------------------
TEST_CASE("empty_graph") {
    Graph G;                                      // no vertices
    auto m = minimal_fraction_max_matching(G);
    REQUIRE(m.empty());
}

// ------------------------------ 2  Single edge ------------------------------
TEST_CASE("single_edge") {
    Graph G(3);
    add_edge(1, 2, G);
    auto m = minimal_fraction_max_matching(G);
    REQUIRE(m.size() == 1);
    REQUIRE(m.at({1,2}) == Approx(1.0));
}

// ------------------------------ 3  Triangle ------------------------------
TEST_CASE("triangle_cycle") {
    Graph G(3);
    add_edge(0,1,G); add_edge(1,2,G); add_edge(0,2,G);

    auto m = minimal_fraction_max_matching(G);
    for (const auto& [e, val] : m) REQUIRE(val == Approx(0.5));

    for (Vertex v = 0; v < 3; ++v)
        REQUIRE(incident_sum(v, m) == Approx(1.0));
}

// ------- 4  4-cycle: optimal should be two ½-edges (weight 2 total) -------
TEST_CASE("square_min_half_edges") {
    Graph G(4);
    add_edge(0,1,G); add_edge(1,2,G);
    add_edge(2,3,G); add_edge(3,0,G);

    auto m = minimal_fraction_max_matching(G);
    double total = 0.0;
    for (const auto& [_, val] : m) total += val;
    REQUIRE(total == Approx(2.0));

    for (Vertex v = 0; v < 4; ++v)
        REQUIRE(incident_sum(v, m) <= 1.0 + 1e-12);
}

// ------------------------------ 5  Path 0-1-2-3 -----------------------------
TEST_CASE("path_length_3") {
    Graph G(4);
    add_edge(0,1,G); add_edge(1,2,G); add_edge(2,3,G);

    auto m = minimal_fraction_max_matching(G);
    double total = 0.0;
    for (const auto& [_, val] : m) total += val;
    REQUIRE(total == Approx(2.0));

    for (Vertex v = 0; v < 4; ++v)
        REQUIRE(incident_sum(v, m) <= 1.0 + 1e-12);
}

// ------------------------------ 6  6-cycle ------------------------------
TEST_CASE("cycle_even") {
    Graph G(6);
    for (int i = 0; i < 6; ++i) add_edge(i, (i+1)%6, G);

    auto m = minimal_fraction_max_matching(G);
    double total = 0.0;
    for (const auto& [_, val] : m) total += val;
    REQUIRE(total == Approx(3.0));

    for (Vertex v = 0; v < 6; ++v)
        REQUIRE(incident_sum(v, m) == Approx(1.0));
}

// ------------------------------ 7  K3,3 ------------------------------
TEST_CASE("complete_bipartite_k33") {
    Graph G(6);
    for (int u = 0; u < 3; ++u)
        for (int v = 3; v < 6; ++v)
            add_edge(u, v, G);

    auto m = minimal_fraction_max_matching(G);
    double total = 0.0;
    for (const auto& [_, val] : m) total += val;
    REQUIRE(total == Approx(3.0));
}

// ------------------------------ 8  K4 ------------------------------
TEST_CASE("complete_graph_k4") {
    Graph G(4);
    for (int u = 0; u < 4; ++u)
        for (int v = u+1; v < 4; ++v)
            add_edge(u, v, G);

    auto m = minimal_fraction_max_matching(G);
    double total = 0.0;
    for (const auto& [_, val] : m) total += val;
    REQUIRE(total == Approx(2.0));

    for (Vertex v = 0; v < 4; ++v)
        REQUIRE(incident_sum(v, m) == Approx(1.0));
}

// ------------- 9  small random graphs: constraints only --------------------
TEST_CASE("random_small_constraints") {
    boost::mt19937 rng(42);
    for (int iter = 0; iter < 10; ++iter) {
        Graph G;
        boost::generate_random_graph(G, 5, 0.5, rng, false);

        auto m = minimal_fraction_max_matching(G);
        for (Vertex v = 0; v < boost::num_vertices(G); ++v)
            REQUIRE(incident_sum(v, m) <= 1.0 + 1e-12);

        for (const auto& [_, val] : m)
            REQUIRE( (val==Approx(0.5) || val==Approx(1.0)) );
    }
}

// ------------- 10  moderate random graphs: feasibility ---------------------
TEST_CASE("large_random_constraints") {
    struct Param { int n; double p; };
    const Param cases[]{{50,0.08},{120,0.04},{250,0.02}};
    boost::mt19937 rng(42);

    for (auto [n,p] : cases) {
        Graph G;
        boost::generate_random_graph(G, n,
            static_cast<std::size_t>(p*n*(n-1)/2), rng, false);

        auto m = minimal_fraction_max_matching(G);
        std::vector<double> load(n, 0.0);
        for (const auto& [e, val] : m) {
            REQUIRE( (val==Approx(0.5) || val==Approx(1.0)) );
            load[e.u] += val; load[e.v] += val;
        }
        for (double w : load)
            REQUIRE(w <= 1.0 + 1e-12);
    }
}
