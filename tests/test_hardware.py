import pytest

from qroute.hardware import HardwareGraph


class TestConstruction:
    def test_line(self):
        hw = HardwareGraph.from_line(4)
        assert hw.num_qubits == 4
        assert hw.are_adjacent(0, 1)
        assert not hw.are_adjacent(0, 2)

    def test_ring(self):
        hw = HardwareGraph.from_ring(4)
        assert hw.are_adjacent(0, 3)
        assert hw.are_adjacent(3, 0)

    def test_ring_too_small_raises(self):
        with pytest.raises(ValueError):
            HardwareGraph.from_ring(2)

    def test_grid_dimensions_and_adjacency(self):
        hw = HardwareGraph.from_grid(2, 3)
        assert hw.num_qubits == 6
        # (0,0)=0 (0,1)=1 (0,2)=2
        # (1,0)=3 (1,1)=4 (1,2)=5
        assert hw.are_adjacent(0, 1)
        assert hw.are_adjacent(0, 3)
        assert not hw.are_adjacent(0, 4)
        assert not hw.are_adjacent(0, 2)

    def test_complete_graph_all_adjacent(self):
        hw = HardwareGraph.from_complete(5)
        for a in range(5):
            for b in range(5):
                if a != b:
                    assert hw.are_adjacent(a, b)

    def test_from_edge_list(self):
        hw = HardwareGraph.from_edge_list(3, [(0, 1), (1, 2)])
        assert hw.are_adjacent(0, 1)
        assert not hw.are_adjacent(0, 2)

    def test_random_sparse_is_connected(self):
        hw = HardwareGraph.from_random_sparse(8, 0.4, seed=1)
        assert hw.is_connected()

    def test_non_contiguous_nodes_rejected(self):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from([0, 1, 3])
        g.add_edge(0, 1)
        with pytest.raises(ValueError):
            HardwareGraph(g)


class TestQueries:
    def test_neighbors(self):
        hw = HardwareGraph.from_line(4)
        assert set(hw.neighbors(1)) == {0, 2}
        assert set(hw.neighbors(0)) == {1}

    def test_shortest_path(self):
        hw = HardwareGraph.from_line(5)
        assert hw.shortest_path(0, 4) == [0, 1, 2, 3, 4]

    def test_distance(self):
        hw = HardwareGraph.from_line(5)
        assert hw.distance(0, 4) == 4
        assert hw.distance(2, 2) == 0

    def test_all_distances_symmetric(self):
        hw = HardwareGraph.from_ring(6)
        d = hw.all_distances()
        assert d[0][3] == d[3][0]

    def test_diameter(self):
        hw = HardwareGraph.from_line(5)
        assert hw.diameter() == 4

    def test_degree_and_degrees(self):
        hw = HardwareGraph.from_grid(2, 2)
        assert hw.degree(0) == 2
        degrees = hw.degrees()
        assert degrees[0] == 2

    def test_edges(self):
        hw = HardwareGraph.from_line(3)
        assert set(hw.edges()) == {(0, 1), (1, 2)}
