// A* pathfinding on a 3D grid (6-connected), implemented from scratch.
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Node {
    int x, y, z;
    double f;  // f = g + h
};

// Minimal binary min-heap keyed on Node::f.
class MinHeap {
    std::vector<Node> a;
public:
    bool empty() const { return a.empty(); }
    void push(const Node& n) {
        a.push_back(n);
        size_t i = a.size() - 1;
        while (i > 0) {
            size_t p = (i - 1) / 2;
            if (a[p].f <= a[i].f) break;
            std::swap(a[p], a[i]);
            i = p;
        }
    }
    Node pop() {
        Node top = a[0];
        a[0] = a.back();
        a.pop_back();
        size_t i = 0, n = a.size();
        while (true) {
            size_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < n && a[l].f < a[m].f) m = l;
            if (r < n && a[r].f < a[m].f) m = r;
            if (m == i) break;
            std::swap(a[i], a[m]);
            i = m;
        }
        return top;
    }
};

struct Pos { int x, y, z; };

// 3D Manhattan distance: admissible and consistent for 6-connected unit-cost grids.
static double heuristic(const Pos& a, const Pos& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
}

// grid[z][y][x]: '#' = wall, anything else walkable. Returns path from start to
// goal (inclusive), or an empty vector if unreachable.
std::vector<Pos> aStar(const std::vector<std::vector<std::string>>& grid,
                       Pos start, Pos goal) {
    int D = grid.size(), H = D ? grid[0].size() : 0, W = H ? grid[0][0].size() : 0;
    auto in = [&](const Pos& p) { return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < W && p.y < H && p.z < D; };
    if (!in(start) || !in(goal) || grid[start.z][start.y][start.x] == '#' ||
        grid[goal.z][goal.y][goal.x] == '#')
        return {};
    auto idx = [&](const Pos& p) { return ((size_t)p.z * H + p.y) * W + p.x; };
    const double INF = std::numeric_limits<double>::infinity();
    size_t V = (size_t)D * H * W;
    std::vector<double> g(V, INF);
    std::vector<long long> parent(V, -1);
    std::vector<bool> closed(V, false);

    MinHeap open;
    g[idx(start)] = 0;
    open.push({start.x, start.y, start.z, heuristic(start, goal)});

    const int dx[] = {1, -1, 0, 0, 0, 0}, dy[] = {0, 0, 1, -1, 0, 0}, dz[] = {0, 0, 0, 0, 1, -1};
    while (!open.empty()) {
        Node cur = open.pop();
        Pos cp{cur.x, cur.y, cur.z};
        if (closed[idx(cp)]) continue;  // stale duplicate entry
        closed[idx(cp)] = true;

        if (cp.x == goal.x && cp.y == goal.y && cp.z == goal.z) {
            std::vector<Pos> path;
            for (long long i = (long long)idx(goal); i != -1; i = parent[i])
                path.push_back({(int)(i % W), (int)((i / W) % H), (int)(i / ((long long)W * H))});
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (int d = 0; d < 6; ++d) {
            Pos np{cp.x + dx[d], cp.y + dy[d], cp.z + dz[d]};
            if (!in(np) || grid[np.z][np.y][np.x] == '#' || closed[idx(np)]) continue;
            double ng = g[idx(cp)] + 1;
            if (ng < g[idx(np)]) {
                g[idx(np)] = ng;
                parent[idx(np)] = (long long)idx(cp);
                open.push({np.x, np.y, np.z, ng + heuristic(np, goal)});
            }
        }
    }
    return {};
}

int main() {
    // Layers indexed by z (printed one after another); each layer is rows of x.
    std::vector<std::vector<std::string>> grid = {
        {   // z = 0
            "S..#.",
            ".#.#.",
            ".#...",
            ".####",
            ".....",
        },
        {   // z = 1
            "###.#",
            "###.#",
            "###..",
            "#####",
            "####.",
        },
        {   // z = 2
            ".....",
            ".###.",
            ".#...",
            ".#.##",
            "...#G",
        },
    };
    Pos start{0, 0, 0}, goal{4, 4, 2};
    auto path = aStar(grid, start, goal);
    if (path.empty()) { std::cout << "No path found\n"; return 1; }
    for (auto& p : path)
        if (grid[p.z][p.y][p.x] == '.') grid[p.z][p.y][p.x] = '*';
    for (size_t z = 0; z < grid.size(); ++z) {
        std::cout << "z = " << z << "\n";
        for (auto& row : grid[z]) std::cout << row << "\n";
    }
    std::cout << "Path length: " << path.size() - 1 << " steps\n";
}
