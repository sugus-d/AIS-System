"""mesh.roi.bfs — 面值 → 顶点散射（共享函数）测试。"""

from __future__ import annotations

import numpy as np

from mesh.roi.bfs import scatter_face_values_to_vertices


def test_scatter_shared_vertex_averages_incident_faces():
    """共享顶点取关联面的均值（顶点1 关联两个面 → (2+4)/2=3）。"""
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    values = np.array([2.0, 4.0])
    out = scatter_face_values_to_vertices(faces, values, 4)
    assert np.allclose(out, [2.0, 3.0, 3.0, 4.0])


def test_unreferenced_vertices_stay_zero():
    """未关联任何面的顶点保持 0（除零保护）。"""
    faces = np.array([[0, 1, 2]])
    values = np.array([5.0])
    out = scatter_face_values_to_vertices(faces, values, 6)
    assert np.allclose(out, [5.0, 5.0, 5.0, 0.0, 0.0, 0.0])


def test_single_face_constant_value():
    """单面常数值 → 该面 3 顶点均为该值。"""
    faces = np.array([[2, 5, 7]])
    values = np.array([3.0])
    out = scatter_face_values_to_vertices(faces, values, 8)
    assert out[2] == out[5] == out[7] == 3.0
    assert out.sum() == 9.0


def test_empty_faces_returns_zeros():
    """无面输入返回全 0（不崩溃）。"""
    out = scatter_face_values_to_vertices(np.empty((0, 3), dtype=int), np.empty(0), 4)
    assert np.allclose(out, np.zeros(4))
