// A* pathfinding on a 2D grid (4-connected), implemented from scratch.
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Node {
    int x, y;
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

// Manhattan distance: admissible and consistent for 4-connected unit-cost grids.
static double heuristic(int x, int y, int gx, int gy) {
    return std::abs(x - gx) + std::abs(y - gy);
}

// grid: '#' = wall, anything else walkable. Returns path from start to goal
// (inclusive), or an empty vector if unreachable.
std::vector<std::pair<int,int>> aStar(const std::vector<std::string>& grid,
                                      std::pair<int,int> start,
                                      std::pair<int,int> goal) {
    int H = grid.size(), W = H ? grid[0].size() : 0;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> g(H, std::vector<double>(W, INF));
    std::vector<std::vector<std::pair<int,int>>> parent(
        H, std::vector<std::pair<int,int>>(W, {-1, -1}));
    std::vector<std::vector<bool>> closed(H, std::vector<bool>(W, false));

    MinHeap open;
    g[start.second][start.first] = 0;
    open.push({start.first, start.second,
               heuristic(start.first, start.second, goal.first, goal.second)});

    const int dx[] = {1, -1, 0, 0}, dy[] = {0, 0, 1, -1};
    while (!open.empty()) {
        Node cur = open.pop();
        if (closed[cur.y][cur.x]) continue;  // stale duplicate entry
        closed[cur.y][cur.x] = true;

        if (cur.x == goal.first && cur.y == goal.second) {
            std::vector<std::pair<int,int>> path;
            for (std::pair<int,int> p = goal; p.first != -1; p = parent[p.second][p.first])
                path.push_back(p);
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (int d = 0; d < 4; ++d) {
            int nx = cur.x + dx[d], ny = cur.y + dy[d];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            if (grid[ny][nx] == '#' || closed[ny][nx]) continue;
            double ng = g[cur.y][cur.x] + 1;
            if (ng < g[ny][nx]) {
                g[ny][nx] = ng;
                parent[ny][nx] = {cur.x, cur.y};
                open.push({nx, ny, ng + heuristic(nx, ny, goal.first, goal.second)});
            }
        }
    }
    return {};
}

int main() {
    std::vector<std::string> grid = {
        "S...#.....",
        ".##.#.###.",
        ".#..#...#.",
        ".#.####.#.",
        ".#......#.",
        ".######.#.",
        "........#G",
    };
    std::pair<int,int> start{0, 0}, goal{9, 6};
    auto path = aStar(grid, start, goal);
    if (path.empty()) { std::cout << "No path found\n"; return 1; }
    for (auto& p : path)
        if (grid[p.second][p.first] == '.') grid[p.second][p.first] = '*';
    for (auto& row : grid) std::cout << row << "\n";
    std::cout << "Path length: " << path.size() - 1 << " steps\n";
}
