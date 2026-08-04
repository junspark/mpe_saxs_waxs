"""Ring table builder — thin wrapper over midas_hkls + per-detector geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from midas_hkls import Lattice, SpaceGroup, generate_hkls

from .params import CalibrationParams


@dataclass
class RingTable:
    ring_nr: np.ndarray         # [n_rings] int
    h: np.ndarray
    k: np.ndarray
    l: np.ndarray
    d_spacing: np.ndarray       # Å
    two_theta_deg: np.ndarray
    multiplicity: np.ndarray    # int
    r_ideal_px: np.ndarray      # px

    def __len__(self) -> int:
        return len(self.ring_nr)


def _build_ring_table_from_dspacings(params: CalibrationParams, px: float,
                                      two_theta_max: float) -> RingTable:
    """Lamellar-calibrant path (e.g. silver behenate): derive rings from
    user-supplied d-spacings via Bragg's law, mirroring
    CalibrantIntegratorOMP.c's DSpacing bypass of GetHKLList. Ring numbers
    are assigned 1..N in the order given. h/k/l/multiplicity have no
    meaning for a non-crystalline calibrant and are left as placeholders
    (0/0/0/1) — nothing downstream reads them for anything but display."""
    ring_nr: List[int] = []
    d_sp: List[float] = []
    tth: List[float] = []
    for i, d in enumerate(params.DSpacings):
        if d <= 0:
            continue
        ratio = params.Wavelength / (2.0 * d)
        if abs(ratio) >= 1.0:
            continue
        theta_deg = np.degrees(np.arcsin(ratio))
        if 2.0 * theta_deg > two_theta_max:
            continue
        ring_nr.append(i + 1)
        d_sp.append(d)
        tth.append(2.0 * theta_deg)
    if not ring_nr:
        raise RuntimeError(
            "No rings retained from DSpacings — check wavelength / MaxRingRad")

    n = len(ring_nr)
    rt = RingTable(
        ring_nr=np.array(ring_nr, dtype=np.int32),
        h=np.zeros(n, dtype=np.int32),
        k=np.zeros(n, dtype=np.int32),
        l=np.zeros(n, dtype=np.int32),
        d_spacing=np.array(d_sp, dtype=np.float64),
        two_theta_deg=np.array(tth, dtype=np.float64),
        multiplicity=np.ones(n, dtype=np.int32),
        r_ideal_px=np.empty(n, dtype=np.float64),
    )
    rt.r_ideal_px[:] = params.Lsd * np.tan(np.radians(rt.two_theta_deg)) / px
    return rt


def build_ring_table(params: CalibrationParams) -> RingTable:
    """Generate the calibrant ring table at the current detector geometry."""
    # Maximum 2θ from MaxRingRad and Lsd (use mean pixel size).
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    max_R_um = params.MaxRingRad * px
    two_theta_max = 2.0 * np.degrees(np.arctan(max_R_um / params.Lsd))

    if params.DSpacings:
        rt = _build_ring_table_from_dspacings(params, px, two_theta_max)
    else:
        sg = SpaceGroup.from_number(params.SpaceGroup)
        lat = Lattice(*params.LatticeConstant)

        refs = generate_hkls(
            sg, lat,
            wavelength_A=params.Wavelength,
            two_theta_max_deg=two_theta_max,
        )
        if not refs:
            raise RuntimeError("No reflections within max 2θ — check geometry / lattice / wavelength")

        n = len(refs)
        rt = RingTable(
            ring_nr=np.array([r.ring_nr for r in refs], dtype=np.int32),
            h=np.array([r.h for r in refs], dtype=np.int32),
            k=np.array([r.k for r in refs], dtype=np.int32),
            l=np.array([r.l for r in refs], dtype=np.int32),
            d_spacing=np.array([r.d_spacing for r in refs], dtype=np.float64),
            two_theta_deg=np.array([r.two_theta_deg for r in refs], dtype=np.float64),
            multiplicity=np.array([r.multiplicity for r in refs], dtype=np.int32),
            r_ideal_px=np.empty(n, dtype=np.float64),
        )
        rt.r_ideal_px[:] = params.Lsd * np.tan(np.radians(rt.two_theta_deg)) / px

    if params.MinRingRad > 0:
        keep = rt.r_ideal_px >= params.MinRingRad
        for f in ("ring_nr", "h", "k", "l", "d_spacing", "two_theta_deg", "multiplicity", "r_ideal_px"):
            setattr(rt, f, getattr(rt, f)[keep])
    return rt
