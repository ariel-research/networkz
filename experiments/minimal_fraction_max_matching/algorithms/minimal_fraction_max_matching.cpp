// Faithful C++17 port of Roi Sibony’s Python implementation
// Optimised for performance by using Cpp language
// Compile:  g++ -std=c++17 -O2 minimal_fraction_max_matching.cpp

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <fstream>

// ────────────────────────── edge key ────────────────────────────────────────
struct EdgeKey {
    std::uint32_t u{}, v{};
    EdgeKey(std::uint32_t a, std::uint32_t b) {
        if (a < b) { u = a; v = b; }
        else        { u = b; v = a; }
    }
    bool operator==(const EdgeKey& other) const noexcept {
        return u == other.u && v == other.v;
    }
};
struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const noexcept {
        return (static_cast<std::size_t>(e.u) << 32) ^ e.v;
    }
};

// ───────────────────── graph typedefs ───────────────────────────────────────
using Graph  = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

template<typename G>
auto neighbours(Vertex u, const G& g) {
    return boost::make_iterator_range(boost::adjacent_vertices(u, g));
}

// ──────────────────── FractionalMatchingSolver ──────────────────────────────
class FractionalMatchingSolver {
public:
    explicit FractionalMatchingSolver(const Graph& g) : G(g) {
        std::size_t n = boost::num_vertices(G);
        labels.assign(n, Label::None);
        preds .assign(n, kNoPred);
    }

    using Matching =
        std::unordered_map<EdgeKey, double, EdgeKeyHash>;   // (u,v) ↦ 0,0.5,1

    Matching solve() {
        while(true) {
            init_labels();                    // Step 1
            bool augmented = false;

            while (!augmented) {
                auto opt = edge_scan();       // Step 2
                if (!opt) break;

                auto [u, v] = *opt;
                if (labels[v] == Label::Plus) {
                    augment(u, v);            // type 1/3
                    augmented = true;
                } else {
                    if (label_or_augment(u, v)) {
                        augment(u, v);        // type 2
                        augmented = true;
                    }
                }
            }
            if (!augmented) break;
        }
        return final_matching();
    }

private:
    enum class Label : char { Plus = '+', Minus = '-', None = 0 };
    static constexpr int kNoPred = -1;

    const Graph& G;
    Matching     x;            // edge → value
    std::vector<Label> labels; // vertex labels
    std::vector<int>   preds;  // predecessor pointers

    // edge helpers -----------------------------------------------------------
    double get_val(std::uint32_t a, std::uint32_t b) const {
        auto it = x.find(EdgeKey(a, b));
        return it == x.end() ? 0.0 : it->second;
    }
    void   set_val(std::uint32_t a, std::uint32_t b, double val) {
        x[EdgeKey(a, b)] = val;
    }
    void   flip_val(std::uint32_t a, std::uint32_t b) {
        set_val(a, b, 1.0 - get_val(a, b));
    }

    // Step 1 -----------------------------------------------------------------
    void init_labels() {
        std::size_t n = boost::num_vertices(G);
        std::vector<double> sat(n, 0.0);

        for (const auto& [e, val] : x) {
            sat[e.u] += val;
            sat[e.v] += val;
        }
        for (std::size_t v = 0; v < n; ++v) {
            labels[v] = (sat[v] < 1.0) ? Label::Plus : Label::None;
            preds [v] = kNoPred;
        }
    }

    // Step 2 -----------------------------------------------------------------
    std::optional<std::pair<Vertex, Vertex>> edge_scan() {
        for (Vertex u = 0; u < boost::num_vertices(G); ++u) {
            if (labels[u] != Label::Plus) continue;

            for (Vertex v : neighbours(u, G)) {
                if (labels[v] == Label::Minus) continue;
                if (labels[v] == Label::Plus)  return {{u, v}};
                if (labels[v] == Label::None && get_val(u, v) < 1.0)
                    return {{u, v}};
            }
        }
        return std::nullopt;
    }

    // helper: trace to root ---------------------------------------------------
    std::vector<Vertex> trace(Vertex start) const {
        std::vector<Vertex> path{start};
        int cur = preds[start];
        while (cur != kNoPred) {
            path.push_back(static_cast<Vertex>(cur));
            cur = preds[cur];
        }
        return path;
    }

    // build_cycle -------------------------------------------------------
    static std::vector<Vertex>
    build_cycle(const std::vector<Vertex>& path_u,
                const std::vector<Vertex>& path_v)
    {
        std::unordered_set<Vertex> set_u(path_u.begin(), path_u.end());
        auto it  = std::find_if(path_v.begin(), path_v.end(),
                                [&](Vertex x){ return set_u.count(x); });
        Vertex lca = *it;

        std::size_t idx_u = std::find(path_u.begin(), path_u.end(), lca) - path_u.begin();
        std::size_t idx_v = it - path_v.begin();

        std::vector<Vertex> cycle;
        cycle.insert(cycle.end(), path_u.begin(), path_u.begin() + idx_u + 1);

        if (idx_v > 0) {                                 // guard against idx_v==0
            for (auto rit = path_v.begin() + idx_v - 1; ; --rit) {
                cycle.push_back(*rit);
                if (rit == path_v.begin()) break;
            }
        }
        return cycle;
    }

    // Step 3 ------------------------------------------------------------------
    void augment(Vertex u, Vertex v) {
        auto path_u = trace(u);
        auto path_v = trace(v);

        if (path_u.back() != path_v.back()) { // type 1
            std::reverse(path_u.begin(), path_u.end());
            for (std::size_t i = 0; i + 1 < path_u.size(); ++i)
                flip_val(path_u[i], path_u[i + 1]);

            flip_val(u, v);

            for (std::size_t i = 0; i + 1 < path_v.size(); ++i)
                flip_val(path_v[i], path_v[i + 1]);
        }
        else {                                // type 3
            auto cycle = build_cycle(path_u, path_v);
            for (std::size_t i = 0; i < cycle.size(); ++i) {
                Vertex a = cycle[i];
                Vertex b = cycle[(i + 1) % cycle.size()];
                double now = get_val(a, b);
                set_val(a, b, now != 0.5 ? 0.5 : 0.0);
            }
            std::reverse(path_v.begin(), path_v.end());
            for (std::size_t i = 0; i + 1 < path_v.size(); ++i)
                flip_val(path_v[i], path_v[i + 1]);
        }
    }

    // Step 4 ------------------------------------------------------------------
    bool label_or_augment(Vertex u, Vertex v) {
        for (Vertex w : neighbours(v, G))
            if (get_val(v, w) == 1.0) {       // re-label
                labels[v] = Label::Minus;
                labels[w] = Label::Plus;
                preds [v] = static_cast<int>(u);
                preds [w] = static_cast<int>(v);
                return false;
            }
        return type2_from(v);                 // augment on ½-cycle
    }

    bool type2_from(Vertex v) {
        std::vector<Vertex> cycle{v};
        std::unordered_set<Vertex> seen{v};

        for (;;) {
            Vertex cur = cycle.back();
            bool advanced = false;

            for (Vertex nxt : neighbours(cur, G))
                if (get_val(cur, nxt) == 0.5 && !seen.count(nxt)) {
                    cycle.push_back(nxt);
                    seen.insert(nxt);
                    advanced = true;
                    break;
                }
            if (advanced) continue;

            for (Vertex nxt : neighbours(cur, G))
                if (nxt == cycle.front() && get_val(cur, nxt) == 0.5) {
                    for (std::size_t i = 0; i < cycle.size(); ++i) {
                        Vertex a = cycle[i];
                        Vertex b = cycle[(i + 1) % cycle.size()];
                        set_val(a, b, (i % 2 == 0) ? 0.0 : 1.0);
                    }
                    return true;
                }
            return false; // theoretical impossibility
        }
    }

    // summarise ---------------------------------------------------------------
    Matching final_matching() const {
        Matching res;
        for (const auto& [e, val] : x)
            if (val > 0.0) res.emplace(e, val);
        return res;
    }
};

// ───────────────────────── wrapper function ────────────────────────────────
FractionalMatchingSolver::Matching
minimal_fraction_max_matching(const Graph& G)
{
    FractionalMatchingSolver solver(G);
    return solver.solve();
}

// ───────────────────────── main function for benchmarking ─────────────────────────
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }

    // Read graph from file
    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << argv[1] << std::endl;
        return 1;
    }

    int n, m;
    infile >> n >> m;

    // Create boost graph
    Graph G(n);
    
    for (int i = 0; i < m; i++) {
        int u, v;
        infile >> u >> v;
        boost::add_edge(u, v, G);
    }

    // Run fractional matching algorithm
    auto matching = minimal_fraction_max_matching(G);
    
    // Calculate matching value
    double value = 0.0;
    for (const auto& [edge, weight] : matching) {
        value += weight;
    }
    
    // Output only the matching value (to be parsed by benchmark.py)
    std::cout << value << std::endl;
    
    return 0;
}