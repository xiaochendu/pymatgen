"""
Tests for temperature support in Pourbaix diagram module.
"""

from __future__ import annotations

import warnings
from unittest import TestCase

import numpy as np
from pytest import approx

from pymatgen.analysis.pourbaix_diagram import (
    TEMPERATURE_DEFAULT,
    KB_EV,
    get_prefac,
    HydrogenPourbaixEntry,
    IonEntry,
    MultiEntry,
    OxygenPourbaixEntry,
    PourbaixDiagram,
    PourbaixEntry,
    SurfacePourbaixEntry,
    PREFAC_DEFAULT,
)
from pymatgen.core import Structure
from pymatgen.core.ion import Ion
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.util.testing import TEST_FILES_DIR, PymatgenTest

TEST_DIR = f"{TEST_FILES_DIR}/analysis/pourbaix_diagram"



# --- BEGIN: Combined temperature inference tests ---
import sys
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry

def test_temperature_inference():
    """Test that PourbaixDiagram can infer temperature from entries."""
    temp1 = 400.0  # K
    temp2 = 400.0  # K
    entry1 = ComputedEntry(Composition("Fe"), -2.0)
    entry2 = ComputedEntry(Composition("FeO"), -3.0)
    pbx_entry1 = PourbaixEntry(entry1, temperature=temp1)
    pbx_entry2 = PourbaixEntry(entry2, temperature=temp2)
    # Test 1: Temperature inference from entries
    diagram = PourbaixDiagram([pbx_entry1, pbx_entry2])
    assert abs(diagram.temperature - temp1) < 1e-6
    # Test 2: Explicit temperature with consistent entries
    diagram2 = PourbaixDiagram([pbx_entry1, pbx_entry2], temperature=temp1)
    assert abs(diagram2.temperature - temp1) < 1e-6
    # Test 3: Inconsistent temperature should raise error
    try:
        pbx_entry3 = PourbaixEntry(ComputedEntry(Composition("Fe2O3"), -4.0), temperature=500.0)
        diagram3 = PourbaixDiagram([pbx_entry1, pbx_entry3])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    # Test 4: No temperature in entries (should use default)
    entry3 = ComputedEntry(Composition("Fe"), -2.0)
    entry4 = ComputedEntry(Composition("FeO"), -3.0)
    pbx_entry3 = PourbaixEntry(entry3)
    pbx_entry4 = PourbaixEntry(entry4)
    diagram4 = PourbaixDiagram([pbx_entry3, pbx_entry4])
    assert abs(diagram4.temperature - 298.15) < 1e-6
# --- END: Combined temperature inference tests ---