import networkx as nx


def create_graph_from_grid(grid):
    graph = nx.Graph()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                continue

            graph.add_node((i, j))

            if i > 0 and grid[i - 1][j] != 0:
                graph.add_edge((i, j), (i - 1, j))
            if i < rows - 1 and grid[i + 1][j] != 0:
                graph.add_edge((i, j), (i + 1, j))
            if j > 0 and grid[i][j - 1] != 0:
                graph.add_edge((i, j), (i, j - 1))
            if j < cols - 1 and grid[i][j + 1] != 0:
                graph.add_edge((i, j), (i, j + 1))

            if i > 0 and j > 0 and grid[i - 1][j - 1] != 0:
                graph.add_edge((i, j), (i - 1, j - 1))
            if i > 0 and j < cols - 1 and grid[i - 1][j + 1] != 0:
                graph.add_edge((i, j), (i - 1, j + 1))
            if i < rows - 1 and j > 0 and grid[i + 1][j - 1] != 0:
                graph.add_edge((i, j), (i + 1, j - 1))
            if i < rows - 1 and j < cols - 1 and grid[i + 1][j + 1] != 0:
                graph.add_edge((i, j), (i + 1, j + 1))

    return graph


def update_graph(graph, grid, changed_cells):
    for cell in changed_cells:
        i, j = cell
        if grid[i][j] == 0:
            if graph.has_node(cell):
                graph.remove_node(cell)
        else:
            if not graph.has_node(cell):
                graph.add_node(cell)

            for di in range(-1, 2):  # -1, 0, 1
                for dj in range(-1, 2):  # -1, 0, 1
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj
                    if grid.shape[0] > ni >= 0 != grid[ni][nj] and 0 <= nj < grid.shape[1]:
                        graph.add_edge(cell, (ni, nj))

    return graph
