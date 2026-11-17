# A* Navigation

`astar.cpp` implements A* from scratch (including its own binary min-heap) on a 6-connected 3D grid (moves along ±x, ±y, ±z).

Build and run:

    g++ -std=c++17 -O2 -o astar astar.cpp && ./astar

## How A* works

A* finds a shortest path from a start to a goal by repeatedly expanding the most promising node. Each node `n` has:

- `g(n)`: cost of the best known path from start to `n`.
- `h(n)`: heuristic estimate of the cost from `n` to the goal.
- `f(n) = g(n) + h(n)`: estimated total cost of a path through `n`.

Algorithm:

1. Put the start in the *open set* (a priority queue ordered by `f`) with `g = 0`.
2. Pop the node with the lowest `f`. If it is already closed, skip it. Otherwise close it.
3. If it is the goal, rebuild the path by following parent links back to the start.
4. For each walkable neighbour, compute `g' = g(current) + step cost`. If `g'` beats the neighbour's best known `g`, record the new `g`, set its parent to the current node, and push it with `f = g' + h`.
5. If the open set empties, no path exists.

With an **admissible** heuristic (never overestimates) the first path found to the goal is optimal. With a **consistent** one, a closed node never needs reopening. Here the heuristic is 3D Manhattan distance (`|dx|+|dy|+|dz|`), which is both for 6-directional movement with unit cost. With `h = 0`, A* becomes Dijkstra's algorithm. A larger `h` makes it greedier and faster, but it can lose optimality.

Implementation notes:

- The heap has no decrease-key. Improved nodes are pushed again, and stale entries are skipped via the closed set.
- Complexity is O(V log V) for a grid with V cells in the worst case. A good heuristic usually expands far fewer.
- The grid is indexed `grid[z][y][x]`: a vector of layers, each a vector of row strings. Positions are `Pos{x, y, z}`; out-of-range or walled start/goal return no path.
- The demo grid (three layers, printed per `z`) uses `#` for walls, `S` for the start, `G` for the goal, and `*` for the path.
