.. _edge_layout:

Edge Layout / Routing
=====================

.. currentmodule:: netgraph

.. autosummary::

    get_straight_edge_paths
    get_curved_edge_paths
    get_bundled_edge_paths

.. autofunction:: get_straight_edge_paths
.. autofunction:: get_curved_edge_paths
.. autofunction:: get_bundled_edge_paths

``get_bundled_edge_paths`` accepts an optional ``processes`` argument to
parallelise the computation of edge compatibilities and forces.  A separate
``ProcessPoolExecutor`` is created for the duration of the call when this value
is greater than ``1``.

The function automatically falls back to a fully vectorised single-process
implementation when ``processes`` is ``None`` or ``1``.  Because the
vectorisation removes the need for Python-level loops, this code path tends to
be faster for small or medium graphs where the overhead of inter-process
communication outweighs the benefits of parallelisation.  For larger graphs you
may obtain speedups by increasing ``processes`` up to the number of available
CPU cores.
