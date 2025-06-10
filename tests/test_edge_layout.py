#!/usr/bin/env python
"""
Test _edge_layout.py
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from netgraph._main import Graph
from netgraph._edge_layout import (
    get_bundled_edge_paths,
    _get_edge_compatibility,
    _initialize_bundled_control_points,
    _get_Fe,
)
from netgraph._utils import _get_point_on_a_circle
from toy_graphs import star

np.random.seed(42)


@pytest.mark.mpl_image_compare
def test_straight_edge_layout():
    edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0)]
    node_positions = {
        0 : (0.2, 0.2),
        1 : (0.5, 0.8),
        2 : (0.8, 0.2),
    }
    fig, ax = plt.subplots()
    Graph(edges, node_layout=node_positions, edge_layout='straight', arrows=True)
    return fig


@pytest.mark.mpl_image_compare
def test_curved_edge_layout():
    fig, ax = plt.subplots()
    edges = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 2),
    ]
    node_positions = {
        0 : np.array([0.1, 0.1]),
        1 : np.array([0.5, 0.5]),
        2 : np.array([0.9, 0.89]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='curved')
    return fig


@pytest.mark.mpl_image_compare
def test_arced_edge_layout():
    fig, ax = plt.subplots()
    edges = [
        (0, 1),
    ]
    node_positions = {
        0 : np.array([0.1, 0.5]),
        1 : np.array([0.9, 0.5])
    }
    Graph(edges, node_layout=node_positions, edge_layout='arc', edge_layout_kwargs=dict(rad=1.))
    return fig


# --------------------------------------------------------------------------------
# bundled edge layout

@pytest.mark.mpl_image_compare
def test_draw_bundled_edges():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3)]
    node_positions = {
        0 : np.array([0, 0.25]),
        1 : np.array([1, 0.25]),
        2 : np.array([0, 0.75]),
        3 : np.array([1, 0.75]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_scale_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([-1.5, 0.75]),
        5 : np.array([ 2.5, 0.75]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-1.6, 2.6, -0.1, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_position_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, -1.0]),
        1 : np.array([ 1.0, -1.0]),
        2 : np.array([ 0.0, 0.0]),
        3 : np.array([ 1.0, 0.0]),
        4 : np.array([ 0.0, 4.0]),
        5 : np.array([ 1.0, 4.0]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-0.1, 1.1, -1.1, 4.1])
    return fig


@pytest.mark.mpl_image_compare
def test_angle_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.25]),
        1 : np.array([ 1.0, 0.25]),
        2 : np.array([ 0.0, 0.50]),
        3 : np.array([ 1.0, 0.50]),
        4 : np.array([ 0.0, 0.55]),
        5 : np.array([ 1.0, 0.95]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_visibility_compatibility():
    fig, ax = plt.subplots()
    edges = [(0, 1), (2, 3), (4, 5)]
    node_positions = {
        0 : np.array([ 0.0, 0.]),
        1 : np.array([ 1.0, 0.]),
        2 : np.array([ 1.0, 1.]),
        3 : np.array([ 2.0, 1.]),
        4 : np.array([ 0.0, -np.sqrt(2)]), # i.e. distance between midpoints from (0, 1) to (2, 3) the same as (0, 1) to (4, 5)
        5 : np.array([ 1.0, -np.sqrt(2)]),
    }
    Graph(edges, node_layout=node_positions, edge_layout='bundled', ax=ax)
    ax.axis([-0.1, 2.1, -1.5, 1.1])
    return fig


@pytest.mark.mpl_image_compare
def test_draw_star_graph_with_bundled_edges():
    fig, ax = plt.subplots()
    total_edges = len(star)
    origin = (0.5, 0.5)
    radius = 0.5
    node_positions = {ii+1 : _get_point_on_a_circle(origin, radius, 2*np.pi*np.random.rand()) for ii in range(total_edges)}
    node_positions[0] = origin
    node_positions = {k : np.array(v) for k, v in node_positions.items()}
    Graph(star, node_layout=node_positions, edge_layout='bundled', ax=ax)
    return fig


def test_bundled_edges_processes_one():
    edges = [(0, 1), (2, 3)]
    node_positions = {
        0: np.array([0, 0.25]),
        1: np.array([1, 0.25]),
        2: np.array([0, 0.75]),
        3: np.array([1, 0.75]),
    }
    paths_serial = get_bundled_edge_paths(edges, node_positions)
    paths_process = get_bundled_edge_paths(edges, node_positions, processes=1)
    for edge in paths_serial:
        assert np.allclose(paths_serial[edge], paths_process[edge])


def test_bundled_edges_processes_two():
    edges = [(0, 1), (1, 2)]
    node_positions = {
        0: np.array([0, 0.25]),
        1: np.array([0.5, 0.75]),
        2: np.array([1, 0.25]),
    }
    paths_serial = get_bundled_edge_paths(edges, node_positions)
    paths_process = get_bundled_edge_paths(edges, node_positions, processes=2)
    for edge in paths_serial:
        assert np.allclose(paths_serial[edge], paths_process[edge])


def _old_edge_compatibility_worker(args):
    """Original loop-based worker from previous implementation."""
    from netgraph._edge_layout import (
        _get_scale_compatibility,
        _get_position_compatibility,
        _get_angle_compatibility,
        _get_visibility_compatibility,
    )
    chunk, edge_to_segment, threshold = args
    out = []
    for e1, e2 in chunk:
        P = edge_to_segment[e1]
        Q = edge_to_segment[e2]

        compatibility = 1
        compatibility *= _get_scale_compatibility(P, Q)
        if compatibility < threshold:
            continue
        compatibility *= _get_position_compatibility(P, Q)
        if compatibility < threshold:
            continue
        compatibility *= _get_angle_compatibility(P, Q)
        if compatibility < threshold:
            continue
        compatibility *= _get_visibility_compatibility(P, Q)
        if compatibility < threshold:
            continue

        reverse = min(np.linalg.norm(P[0] - Q[0]), np.linalg.norm(P[1] - Q[1])) > \
            min(np.linalg.norm(P[0] - Q[1]), np.linalg.norm(P[1] - Q[0]))

        out.append((e1, e2, compatibility, reverse))
    return out


def test_edge_compatibility_worker_vectorised_matches_loop():
    from netgraph._edge_layout import Segment, _edge_compatibility_worker
    import itertools

    node_positions = {i: np.random.rand(2) for i in range(6)}
    edges = [(i, i + 1) for i in range(5)]
    edge_to_segment = {
        e: Segment(node_positions[e[0]], node_positions[e[1]]) for e in edges
    }
    pairs = list(itertools.combinations(edges, 2))
    args = (pairs, edge_to_segment, 0.0)

    expected = _old_edge_compatibility_worker(args)
    result = _edge_compatibility_worker(args)

    assert len(expected) == len(result)
    assert all(
        np.allclose(e[2], r[2]) and (e[0] == r[0]) and (e[1] == r[1]) and (e[3] == r[3])
        for e, r in zip(sorted(expected), sorted(result))
    )
