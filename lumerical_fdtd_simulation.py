"""
Lumerical FDTD backend for NURBS meta-atom simulations.

This module mirrors the Meep-facing interface in ``nurbs_atoms_data.py``:

    sim = LumericalSimulation(control_points)
    transmittance, phase = sim.run_forward(532e-9, 532e-9)

Inputs are SI wavelengths in meters. Control points are in micrometers, matching
the existing project convention. The returned phase is relative to a reference
simulation without the TiO2 meta-atom, and transmittance is normalized by that
same reference when ``normalize=True``.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


DEFAULT_CONTROL_POINTS = np.array(
    [
        (0.18, 0.0),
        (0.16, 0.16),
        (0.0, 0.18),
        (-0.16, 0.16),
        (-0.18, 0.0),
        (-0.16, -0.16),
        (0.0, -0.16),
        (0.16, -0.16),
    ],
    dtype=float,
)


def import_lumapi(lumapi_path: Optional[str] = None):
    """Import Lumerical's lumapi module from an explicit or auto-detected path."""
    candidates = []
    env_path = os.environ.get("LUMERICAL_PYTHON_API")
    if lumapi_path:
        candidates.append(Path(lumapi_path))
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path(r"D:\Program Files\ANSYS Inc\v252\Lumerical\api\python\lumapi.py"),
            Path(r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python\lumapi.py"),
            Path(r"C:\Program Files\Lumerical\v252\api\python\lumapi.py"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("lumapi", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not find lumapi.py. Set LUMERICAL_PYTHON_API to the full path, "
        "for example: D:\\Program Files\\ANSYS Inc\\v252\\Lumerical\\api\\python\\lumapi.py"
    )


def wrap_phase(phase: float) -> float:
    """Wrap phase to [-pi, pi]."""
    return float((phase + np.pi) % (2 * np.pi) - np.pi)


def sort_points_ccw(points: np.ndarray) -> np.ndarray:
    """Sort points counter-clockwise around their centroid."""
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


class LumericalSimulation:
    """Build and run a Lumerical FDTD unit-cell simulation for one NURBS meta-atom."""

    def __init__(
        self,
        control_points: np.ndarray = DEFAULT_CONTROL_POINTS,
        lumapi_path: Optional[str] = None,
        hide: bool = False,
    ):
        self.control_points = np.asarray(control_points, dtype=float)
        self.lumapi = import_lumapi(lumapi_path)
        self.hide = hide
        self.edge_indices = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]]

        self.period = 0.5e-6
        self.cell_z_span = 3.0e-6
        self.substrate_thickness = 1.0e-6
        self.tio2_height = 0.6e-6
        self.transmission_monitor_span = self.period
        self.mesh_accuracy = 2
        self.simulation_time = 1000e-15

        self.source_z = -0.8e-6
        self.monitor_z = 0.8e-6
        self.project_path = None
        self.reference_project_path = None

    def set_control_points(self, control_points: np.ndarray):
        self.control_points = np.asarray(control_points, dtype=float)

    def set_contral_points(self, control_points: np.ndarray):
        """Compatibility with the misspelled Meep method name."""
        self.set_control_points(control_points)

    def generate_complete_nurbs_curve(self, points: np.ndarray) -> np.ndarray:
        all_points = []
        extended_points = np.vstack([points, points[0:1]])

        for edge in self.edge_indices:
            segment_control_points = np.array(
                [
                    extended_points[edge[0]],
                    extended_points[edge[1]],
                    extended_points[edge[2]],
                ]
            )
            all_points.extend(self.generate_nurbs_segment(segment_control_points, num_points=25))

        unique_points = []
        seen = set()
        for point in all_points:
            key = (round(float(point[0]), 6), round(float(point[1]), 6))
            if key not in seen:
                seen.add(key)
                unique_points.append(point)

        return sort_points_ccw(np.asarray(unique_points, dtype=float))

    @staticmethod
    def generate_nurbs_segment(control_points: np.ndarray, num_points: int = 25):
        vertices = []

        def basis_function(index: int, t: float) -> float:
            if index == 0:
                return (1 - t) * (1 - t)
            if index == 1:
                return 2 * t * (1 - t)
            if index == 2:
                return t * t
            return 0.0

        for sample_index in range(num_points):
            t = sample_index / (num_points - 1) if num_points > 1 else 0.0
            x = sum(control_points[i, 0] * basis_function(i, t) for i in range(3))
            y = sum(control_points[i, 1] * basis_function(i, t) for i in range(3))
            vertices.append([x, y])

        return vertices

    def run_forward(
        self,
        wavelength_start: float = 400e-9,
        wavelength_stop: float = 700e-9,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[float, float]:
        wavelength = 0.5 * (float(wavelength_start) + float(wavelength_stop))
        self.project_path = Path(save_path).resolve() if save_path else None
        if self.project_path:
            self.project_path.parent.mkdir(parents=True, exist_ok=True)
            self.reference_project_path = self.project_path.with_name(
                f"{self.project_path.stem}_reference{self.project_path.suffix}"
            )

        structure_result = self._run_single_simulation(wavelength, include_meta_atom=True)

        if normalize:
            reference_result = self._run_single_simulation(wavelength, include_meta_atom=False)
            transmittance = structure_result["transmission"] / max(
                abs(reference_result["transmission"]), 1e-30
            )
            phase = wrap_phase(structure_result["phase"] - reference_result["phase"])
        else:
            transmittance = structure_result["transmission"]
            phase = wrap_phase(structure_result["phase"])

        transmittance = float(np.clip(np.real(transmittance), 0.0, 1.0))

        return transmittance, phase

    def _run_single_simulation(self, wavelength: float, include_meta_atom: bool):
        fdtd = self.lumapi.FDTD(hide=self.hide if include_meta_atom else True)
        try:
            self._build_fdtd(fdtd, wavelength, include_meta_atom)
            self._save_open_project(fdtd, include_meta_atom)
            fdtd.run()
            self._save_open_project(fdtd, include_meta_atom)

            transmission = float(np.real(np.squeeze(fdtd.getresult("T", "T")["T"])))
            field_result = fdtd.getresult("phase", "E")
            ex = np.asarray(field_result["E"])
            ex_value = complex(np.ravel(ex)[0])
            phase = float(np.angle(ex_value))

            self._save_open_project(fdtd, include_meta_atom)
            return {"transmission": transmission, "phase": phase}
        finally:
            if (not include_meta_atom) or self.hide:
                fdtd.close()

    def _build_fdtd(self, fdtd, wavelength: float, include_meta_atom: bool):
        fdtd.addfdtd()
        fdtd.set("dimension", "3D")
        fdtd.set("x span", self.period)
        fdtd.set("y span", self.period)
        fdtd.set("z span", self.cell_z_span)
        fdtd.set("x min bc", "Periodic")
        fdtd.set("x max bc", "Periodic")
        fdtd.set("y min bc", "Periodic")
        fdtd.set("y max bc", "Periodic")
        fdtd.set("z min bc", "PML")
        fdtd.set("z max bc", "PML")
        fdtd.set("mesh accuracy", self.mesh_accuracy)
        fdtd.set("simulation time", self.simulation_time)

        self._add_substrate(fdtd)
        if include_meta_atom:
            self._add_meta_atom(fdtd)
        self._add_source(fdtd, wavelength)
        self._add_monitors(fdtd, wavelength)

    def _add_substrate(self, fdtd):
        fdtd.addrect()
        fdtd.set("name", "SiO2_substrate")
        fdtd.set("x span", self.period)
        fdtd.set("y span", self.period)
        fdtd.set("z min", -self.substrate_thickness)
        fdtd.set("z max", 0.0)
        fdtd.set("material", "SiO2 (Glass) - Palik")

    def _add_meta_atom(self, fdtd):
        vertices_um = self.generate_complete_nurbs_curve(self.control_points)
        vertices_m = vertices_um * 1e-6

        fdtd.addpoly()
        fdtd.set("name", "TiO2_NURBS_meta_atom")
        fdtd.set("vertices", vertices_m)
        fdtd.set("z min", 0.0)
        fdtd.set("z max", self.tio2_height)
        try:
            fdtd.set("material", "TiO2 (Titanium Dioxide) - Palik")
        except Exception:
            fdtd.set("index", math.sqrt(6.25))

    def _add_source(self, fdtd, wavelength: float):
        fdtd.addplane()
        fdtd.set("name", "source")
        fdtd.set("injection axis", "z")
        fdtd.set("direction", "Forward")
        fdtd.set("x span", self.period)
        fdtd.set("y span", self.period)
        fdtd.set("z", self.source_z)
        fdtd.set("wavelength start", wavelength)
        fdtd.set("wavelength stop", wavelength)
        fdtd.set("polarization angle", 0)

    def _add_monitors(self, fdtd, wavelength: float):
        fdtd.addpower()
        fdtd.set("name", "T")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x span", self.transmission_monitor_span)
        fdtd.set("y span", self.transmission_monitor_span)
        fdtd.set("z", self.monitor_z)
        fdtd.set("override global monitor settings", True)
        fdtd.set("frequency points", 1)
        fdtd.set("use source limits", False)
        fdtd.set("wavelength center", wavelength)
        fdtd.set("wavelength span", 0.0)

        fdtd.addprofile()
        fdtd.set("name", "phase")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x span", 0.0)
        fdtd.set("y span", 0.0)
        fdtd.set("z", self.monitor_z)
        fdtd.set("override global monitor settings", True)
        fdtd.set("frequency points", 1)
        fdtd.set("use source limits", False)
        fdtd.set("wavelength center", wavelength)
        fdtd.set("wavelength span", 0.0)

    def _save_open_project(self, fdtd, include_meta_atom: bool):
        project_path = self.project_path if include_meta_atom else self.reference_project_path
        if project_path is None:
            return
        fdtd.save(str(project_path.resolve()))


def main():
    parser = argparse.ArgumentParser(description="Run one Lumerical FDTD NURBS meta-atom simulation.")
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--lumapi-path", type=str, default=None)
    parser.add_argument("--hide", action="store_true", help="Run without showing the Lumerical GUI window.")
    parser.add_argument(
        "--save-fsp",
        type=str,
        default=None,
        help="Optional .fsp path. Defaults to lumerical_runs/nurbs_meta_atom_<wavelength>nm.fsp.",
    )
    args = parser.parse_args()

    default_name = f"nurbs_meta_atom_{args.wavelength_nm:.0f}nm.fsp"
    save_fsp = args.save_fsp or str(Path(__file__).resolve().parent / "lumerical_runs" / default_name)
    Path(save_fsp).resolve().parent.mkdir(parents=True, exist_ok=True)

    sim = LumericalSimulation(lumapi_path=args.lumapi_path, hide=args.hide)
    wavelength = args.wavelength_nm * 1e-9
    transmittance, phase = sim.run_forward(wavelength, wavelength, save_path=save_fsp)

    print(f"wavelength_nm: {args.wavelength_nm:.3f}")
    print(f"project_path: {Path(save_fsp).resolve()}")
    print(f"transmittance: {transmittance:.6f}")
    print(f"phase_rad: {phase:.6f}")


if __name__ == "__main__":
    main()
