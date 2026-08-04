"""DSpacing-based ring generation (lamellar SAXS calibrants, e.g. AgBehenate).

Cross-checks against a real CalibrantIntegratorOMP run on a silver-behenate
frame (402x1024 detector, px=62 um, Lsd=6500000 um), which printed:

    Ring 1: d=58.3800A 2theta=0.1281 deg R=234.4 px
    Ring 2: d=29.1900A 2theta=0.2562 deg R=468.7 px
    Ring 3: d=19.4600A 2theta=0.3842 deg R=703.1 px
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate import CalibrationParams, build_ring_table


def _agbe_params(dspacings=(58.380, 29.190, 19.460)) -> CalibrationParams:
    p = CalibrationParams()
    p.NrPixelsY = 402; p.NrPixelsZ = 1024
    p.pxY = 62.0; p.pxZ = 62.0
    p.Lsd = 6_500_000.0
    # Back-calculated from the screenshot's ring-1 output (d=58.38A,
    # 2theta=0.1281deg) via Bragg's law — reproduces all 3 printed rings to
    # within measurement/rounding tolerance (the displayed "94 keV" CLI arg
    # is itself rounded, so 12.398/94 doesn't quite match).
    p.Wavelength = 0.13052403695277418
    p.SpaceGroup = 0
    p.MaxRingRad = 723.1  # px, matches the RMax printed by the GUI's cake integrator
    p.MinRingRad = 0.0
    p.DSpacings = list(dspacings)
    return p


def test_validate_allows_spacegroup_zero_with_dspacings():
    p = _agbe_params()
    p.validate()  # must not raise


def test_validate_still_rejects_spacegroup_zero_without_dspacings():
    p = _agbe_params(dspacings=())
    with pytest.raises(ValueError, match="SpaceGroup"):
        p.validate()


def test_build_ring_table_from_dspacings_matches_c_binary():
    p = _agbe_params()
    rt = build_ring_table(p)

    assert list(rt.ring_nr) == [1, 2, 3]
    np.testing.assert_allclose(rt.d_spacing, [58.380, 29.190, 19.460], rtol=1e-6)
    np.testing.assert_allclose(rt.two_theta_deg, [0.1281, 0.2562, 0.3842], atol=2e-3)
    np.testing.assert_allclose(rt.r_ideal_px, [234.4, 468.7, 703.1], rtol=2e-3)
    # Placeholders — no crystallographic hkl for a lamellar calibrant.
    assert np.all(rt.h == 0) and np.all(rt.k == 0) and np.all(rt.l == 0)
    assert np.all(rt.multiplicity == 1)


def test_build_ring_table_drops_out_of_range_dspacings():
    p = _agbe_params(dspacings=(58.380, 19.460))
    # Widen MaxRingRad (-> a generous two_theta_max) so only the Bragg
    # impossibility (ratio >= 1) is under test, not the detector's angular
    # cutoff. lambda=40: ring 1 (d=58.38) stays valid (ratio=0.343,
    # 2theta=40.1deg, well inside the ~87deg window); ring 2 (d=19.46)
    # is impossible (ratio=1.028 >= 1) and must be dropped.
    p.MaxRingRad = 100_000.0
    p.Wavelength = 40.0
    rt = build_ring_table(p)
    assert list(rt.ring_nr) == [1]


def test_build_ring_table_still_works_for_crystallographic_path():
    p = CalibrationParams()
    p.NrPixelsY = 2048; p.NrPixelsZ = 2048
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 1000.0
    rt = build_ring_table(p)
    assert len(rt) > 0
    assert not np.all(rt.h == 0)  # real hkl, unlike the DSpacing path
