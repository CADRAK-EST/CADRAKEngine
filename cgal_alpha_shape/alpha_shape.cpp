#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_2.h>

namespace py = pybind11;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
typedef CGAL::Alpha_shape_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2> Alpha_shape_2;
typedef K::Point_2 Point;

std::vector<std::pair<double, double>> compute_alpha_shape(const std::vector<std::pair<double, double>>& points, double alpha) {
    std::vector<Point> pts;
    for (const auto& p : points) {
        pts.emplace_back(p.first, p.second);
    }
    Alpha_shape_2 A(pts.begin(), pts.end(), alpha, Alpha_shape_2::GENERAL);
    std::vector<std::pair<double, double>> result;
    for (auto it = A.alpha_shape_edges_begin(); it != A.alpha_shape_edges_end(); ++it) {
        auto seg = A.segment(*it);
        result.emplace_back(seg.source().x(), seg.source().y());
        result.emplace_back(seg.target().x(), seg.target().y());
    }
    return result;
}

PYBIND11_MODULE(cgal_alpha_shape, m) {
    m.def("compute_alpha_shape", &compute_alpha_shape, "Compute the alpha shape of a set of points");
}