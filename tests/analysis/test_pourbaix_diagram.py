from __future__ import annotations

import logging
import multiprocessing
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from pytest import approx

from pymatgen.analysis.pourbaix_diagram import (
    HydrogenPourbaixEntry,
    IonEntry,
    MultiEntry,
    OxygenPourbaixEntry,
    PourbaixDiagram,
    PourbaixEntry,
    PourbaixPlotter,
    SurfacePourbaixDiagram,
    SurfacePourbaixEntry,
)
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.ion import Ion
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.util.testing import TEST_FILES_DIR, PymatgenTest

np.set_printoptions(precision=5, suppress=True)

TEST_DIR = f"{TEST_FILES_DIR}/analysis/pourbaix_diagram"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestPourbaixEntry(PymatgenTest):
    """Test all functions using a fictitious entry"""

    def setUp(self):
        # comp = Composition("Mn2O3")
        self.sol_entry = ComputedEntry("Mn2O3", 49)
        ion = Ion.from_formula("MnO4-")
        self.ion_entry = IonEntry(ion, 25)
        self.px_ion = PourbaixEntry(self.ion_entry)
        self.px_sol = PourbaixEntry(self.sol_entry)
        self.px_ion.concentration = 1e-4

    def test_pourbaix_entry(self):
        assert self.px_ion.entry.energy == 25, "Wrong Energy!"
        assert self.px_ion.entry.name == "MnO4[-1]", "Wrong Entry!"
        assert self.px_sol.entry.energy == 49, "Wrong Energy!"
        assert self.px_sol.entry.name == "Mn2O3", "Wrong Entry!"
        # assert self.PxIon.energy == 25, "Wrong Energy!"
        # assert self.PxSol.energy == 49, "Wrong Energy!"
        assert self.px_ion.concentration == 1e-4, "Wrong concentration!"

    def test_calc_coeff_terms(self):
        assert self.px_ion.npH == -8, "Wrong npH!"
        assert self.px_ion.nPhi == -7, "Wrong nPhi!"
        assert self.px_ion.nH2O == 4, "Wrong nH2O!"

        assert self.px_sol.npH == -6, "Wrong npH!"
        assert self.px_sol.nPhi == -6, "Wrong nPhi!"
        assert self.px_sol.nH2O == 3, "Wrong nH2O!"

    def test_as_from_dict(self):
        dct = self.px_ion.as_dict()
        ion_entry = self.px_ion.from_dict(dct)
        assert ion_entry.entry.name == "MnO4[-1]", "Wrong Entry!"

        dct = self.px_sol.as_dict()
        sol_entry = self.px_sol.from_dict(dct)
        assert sol_entry.name == "Mn2O3(s)", "Wrong Entry!"
        assert sol_entry.energy == self.px_sol.energy, "as_dict and from_dict energies unequal"

        # Ensure computed entry data persists
        entry = ComputedEntry("TiO2", energy=-20, data={"test": "test"})
        pbx_entry = PourbaixEntry(entry=entry)
        dumpfn(pbx_entry, "pbx_entry.json")
        reloaded = loadfn("pbx_entry.json")
        assert isinstance(reloaded.entry, ComputedEntry)
        assert reloaded.entry.data is not None

    def test_energy_functions(self):
        # TODO: test these for values
        self.px_sol.energy_at_conditions(10, 0)
        self.px_sol.energy_at_conditions(np.array([1, 2, 3]), 0)
        self.px_sol.energy_at_conditions(10, np.array([1, 2, 3]))
        self.px_sol.energy_at_conditions(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_multi_entry(self):
        # TODO: More robust multi-entry test
        m_entry = MultiEntry([self.px_sol, self.px_ion])
        for attr in ["energy", "composition", "nPhi"]:
            assert getattr(m_entry, attr) == getattr(self.px_sol, attr) + getattr(self.px_ion, attr)

        # As dict, from dict
        m_entry_dict = m_entry.as_dict()
        m_entry_new = MultiEntry.from_dict(m_entry_dict)
        assert m_entry_new.energy == m_entry.energy

    def test_multi_entry_repr(self):
        m_entry = MultiEntry([self.px_sol, self.px_ion])
        assert (
            repr(m_entry) == "PourbaixMultiEntry(energy=90.9717, npH=-14.0, nPhi=-13.0, nH2O=7.0, "
            "entry_id=[None, None], species='Mn2O3(s) + MnO4[-1]')"
        )

    def test_get_elt_fraction(self):
        entry = ComputedEntry("Mn2Fe3O3", 49)
        pb_entry = PourbaixEntry(entry)
        assert pb_entry.get_element_fraction("Fe") == approx(0.6)
        assert pb_entry.get_element_fraction("Mn") == approx(0.4)


class TestOxygenPourbaixEntry(PymatgenTest):
    def setUp(self):
        self.entry = ComputedEntry("H8O4", -9.8332)
        self.pbx_entry = OxygenPourbaixEntry(self.entry)

    def test_pourbaix_entry(self):
        assert self.pbx_entry.entry.energy == approx(-9.8332), "Wrong Energy!"
        assert self.pbx_entry.entry.name == "H2O", "Wrong Entry!"
        assert self.pbx_entry.concentration == 1.0, "Wrong concentration!"
        assert self.pbx_entry.phase_type == "Liquid", "Wrong phase type!"
        assert self.pbx_entry.charge == 0, "Wrong charge!"

    def test_calc_coeff_terms(self):
        assert self.pbx_entry.npH == 8.0, "Wrong npH!"
        assert self.pbx_entry.nPhi == 8.0, "Wrong nPhi!"
        assert self.pbx_entry.nH2O == 0.0, "Wrong nH2O!"

        # normalized values
        assert self.pbx_entry.normalized_npH == 2.0, "Wrong normalized npH!"
        assert self.pbx_entry.normalized_nPhi == 2.0, "Wrong normalized nPhi!"
        assert self.pbx_entry.normalized_nH2O == 0.0, "Wrong normalized nH2O!"
        assert self.pbx_entry.normalized_energy == approx(-2.4582998575), "Wrong normalized Energy!"

    def test_as_from_dict(self):
        dct = self.pbx_entry.as_dict()
        liq_entry = self.pbx_entry.from_dict(dct)
        assert liq_entry.name == "H2O(l)", "Wrong Entry!"
        assert liq_entry.energy == self.pbx_entry.energy, "as_dict and from_dict energies unequal"

        # Ensure computed entry data persists
        entry = ComputedEntry("O2", energy=-20, data={"test": "test"})
        pbx_entry = OxygenPourbaixEntry(entry=entry)
        dumpfn(pbx_entry, "pbx_entry.json")
        reloaded = loadfn("pbx_entry.json")
        assert isinstance(reloaded.entry, ComputedEntry)
        assert reloaded.entry.data is not None

    def test_energy_functions(self):
        # when considering the energies in the context of reduction 1/2 O2 or O -> H2O - 2H+ - 2e-
        # the energy at conditions is Gibbs free energy of the reduction (forward) reaction
        # by looking at the Pourbaix diagram for water, these energies should make sense
        assert self.pbx_entry.energy_at_conditions(0, 0) == approx(-9.8332), (
            "Wrong energy at pH = 0 and V = 0!"
        )
        assert self.pbx_entry.energy_at_conditions(14, 0.401) == approx(-0.005999, abs=1e-3), (
            "Wrong energy at pH = 14 and V = 0.401!"
        )
        assert self.pbx_entry.energy_at_conditions(10, 0) == approx(-5.1052), (
            "Wrong energy at pH = 10 and V = 0!"
        )
        assert self.pbx_entry.energy_at_conditions(np.array([1, 2, 3]), 0) == approx(
            [-9.36, -8.888, -8.415], abs=1e-3
        ), "Wrong energy at pH = [1, 2, 3] and V = 0!"
        assert self.pbx_entry.energy_at_conditions(10, np.array([1, 2, 3])) == approx(
            [2.895, 10.895, 18.895], abs=1e-3
        ), "Wrong energy at pH = 10 and V = [1, 2, 3]!"
        assert self.pbx_entry.energy_at_conditions(
            np.array([1, 2, 3]), np.array([1, 2, 3])
        ) == approx(
            [-1.36, 7.112, 15.585],
            abs=1e-3,
        ), "Wrong energy at pH = [1, 2, 3] and V = [1, 2, 3]!"


class TestHydrogenPourbaixEntry(PymatgenTest):
    def setUp(self):
        self.entry = IonEntry(Ion.from_formula("H[1+]"), 0.0)  # standard formation energy
        self.pbx_entry = HydrogenPourbaixEntry(self.entry)

    def test_pourbaix_entry(self):
        assert self.pbx_entry.entry.energy == approx(0.0), "Wrong Energy!"
        assert self.pbx_entry.entry.name == "H[+1]", "Wrong Entry!"
        assert self.pbx_entry.concentration == 1.0, "Wrong concentration!"
        assert self.pbx_entry.phase_type == "Ion", "Wrong phase type!"
        assert self.pbx_entry.charge == 1, "Wrong charge!"

    def test_calc_coeff_terms(self):
        assert self.pbx_entry.npH == -1.0, "Wrong npH!"
        assert self.pbx_entry.nPhi == -1.0, "Wrong nPhi!"
        assert self.pbx_entry.nH2O == 0.0, "Wrong nH2O!"

        # normalized values
        assert self.pbx_entry.normalized_npH == -1.0, "Wrong normalized npH!"
        assert self.pbx_entry.normalized_nPhi == -1.0, "Wrong normalized nPhi!"
        assert self.pbx_entry.normalized_nH2O == 0.0, "Wrong normalized nH2O!"
        assert self.pbx_entry.normalized_energy == approx(0.0), "Wrong normalized Energy!"

    def test_as_from_dict(self):
        dct = self.pbx_entry.as_dict()
        ion_entry = self.pbx_entry.from_dict(dct)
        assert ion_entry.name == "H[+1]", "Wrong Entry!"
        assert ion_entry.energy == self.pbx_entry.energy, "as_dict and from_dict energies unequal"

        # Ensure computed entry data persists
        entry = IonEntry(Ion.from_formula("H3O[+]"), 0.0, attribute="test")
        pbx_entry = HydrogenPourbaixEntry(entry=entry)
        dumpfn(pbx_entry, "pbx_entry.json")
        reloaded = loadfn("pbx_entry.json")
        assert isinstance(reloaded.entry, IonEntry)
        assert reloaded.entry.attribute == "test"

    def test_energy_functions(self):
        # when considering the energies in the context of oxidation (backward) 1/2 H2 or H -> H+ + e-
        # the energy at conditions is the negative of the Gibbs free energy of the reduction (forward) reaction
        # by looking at the Pourbaix diagram for water, these energies should make sense
        assert self.pbx_entry.energy_at_conditions(0, 0) == approx(0.0), (
            "Wrong energy at pH = 0 and V = 0!"
        )
        assert self.pbx_entry.energy_at_conditions(14, -0.829) == approx(0.00159, abs=1e-3), (
            "Wrong energy at pH = 14 and V = -0.829!"
        )
        assert self.pbx_entry.energy_at_conditions(np.array([1, 2, 3]), 0) == approx(
            [-0.0591, -0.1182, -0.1773],
            abs=1e-3,
            # this is the oxidation (reverse) reaction, thus forming H+ becomes more favorable at higher pH
        ), "Wrong energy at pH = [1, 2, 3] and V = 0!"
        assert self.pbx_entry.energy_at_conditions(0, np.array([1, 2, 3])) == approx(
            [-1.0, -2.0, -3.0], abs=1e-3
        ), "Wrong energy at pH = 0 and V = [1, 2, 3]!"
        assert self.pbx_entry.energy_at_conditions(10, np.array([1, 2, 3])) == approx(
            [-1.591, -2.591, -3.591], abs=1e-3
        ), "Wrong energy at pH = 10 and V = [1, 2, 3]!"
        assert self.pbx_entry.energy_at_conditions(
            np.array([1, 2, 3]), np.array([1, 2, 3])
        ) == approx(
            [-1.0591, -2.1182, -3.1773],
            abs=1e-3,
        ), "Wrong energy at pH = [1, 2, 3] and V = [1, 2, 3]!"


class TestSurfacePourbaixEntry(PymatgenTest):
    def setUp(self):
        self.test_data = loadfn(f"{TEST_DIR}/pourbaix_test_data.json")
        # Import cif and create structure
        self.structure = Structure.from_file(f"{TEST_DIR}/SrIrO3_001_2x2x4.cif")
        self.entry = ComputedStructureEntry(self.structure, -133.3124)  # formation energy
        self.reference_entries = {
            "O": OxygenPourbaixEntry(ComputedEntry("H8O4", -9.8332)),
            "Sr": PourbaixEntry(
                IonEntry.from_dict(
                    {"ion": {"Sr": 1.0, "charge": 2.0}, "energy": -5.798066, "name": "Sr[+2]"}
                )
            ),
            "Ir": PourbaixEntry(ComputedEntry("Ir2 O4", -6.2984)),
        }
        self.pbx_entry = SurfacePourbaixEntry(
            self.entry, self.reference_entries, clean_entry=self.entry, clean_entry_factor=1.0
        )

    def test_pourbaix_entry(self):
        assert self.pbx_entry.entry.energy == approx(-133.3124), "Wrong Energy!"
        assert self.pbx_entry.entry.name == "SrIrO3", "Wrong Entry!"
        assert self.pbx_entry.reference_entries["O"].entry.energy == approx(-9.8332), (
            "Wrong Energy!"
        )
        assert self.pbx_entry.reference_entries["O"].entry.name == "H2O", "Wrong Entry!"
        assert self.pbx_entry.reference_entries["Sr"].entry.energy == approx(-5.798066), (
            "Wrong Energy!"
        )
        assert self.pbx_entry.reference_entries["Sr"].entry.name == "Sr[+2]", "Wrong Entry!"
        assert self.pbx_entry.reference_entries["Ir"].entry.energy == approx(-6.2984), (
            "Wrong Energy!"
        )
        assert self.pbx_entry.reference_entries["Ir"].entry.name == "IrO2", "Wrong Entry!"

    def test_calc_coeff_terms(self):
        assert self.pbx_entry.npH == -32.0, "Wrong npH!"
        assert self.pbx_entry.nPhi == 0.0, "Wrong nPhi!"
        assert self.pbx_entry.nH2O == -32.0, "Wrong nH2O!"
        assert self.pbx_entry.energy == approx(54.85056, rel=1e-3), "Wrong Energy!"

        # Normalized values bsaed on clean entry
        assert self.pbx_entry.normalized_npH == -32.0, "Wrong normalized npH!"
        assert self.pbx_entry.normalized_nPhi == 0.0, "Wrong normalized nPhi!"
        assert self.pbx_entry.normalized_nH2O == -32.0, "Wrong normalized nH2O!"
        assert self.pbx_entry.normalized_energy == approx(54.85056, rel=1e-3), (
            "Wrong normalized Energy!"
        )

    def test_as_from_dict(self):
        dct = self.pbx_entry.as_dict()
        loaded_entry = self.pbx_entry.from_dict(dct)
        assert loaded_entry.name == self.pbx_entry.name, "Wrong Entry!"
        assert loaded_entry.energy == self.pbx_entry.energy, (
            "as_dict and from_dict energies unequal"
        )
        assert set(loaded_entry.reference_entries.keys()) == set(self.reference_entries.keys())

        # Ensure reference entries are loaded correctly
        assert loaded_entry.entry.energy == approx(self.pbx_entry.entry.energy), "Wrong Energy!"
        assert loaded_entry.entry.name == self.pbx_entry.entry.name, "Wrong Entry!"
        for key in self.reference_entries.keys():
            assert loaded_entry.reference_entries[key].entry.energy == approx(
                self.reference_entries[key].entry.energy
            ), f"Wrong Energy for {key}!"
            assert (
                loaded_entry.reference_entries[key].entry.name
                == self.reference_entries[key].entry.name
            ), f"Wrong Entry for {key}!"

        # Ensure computed entry data persists
        entry = ComputedEntry("SrIrO3", energy=-20, data={"test": "test"})
        pbx_entry = SurfacePourbaixEntry(entry, self.reference_entries)
        dumpfn(pbx_entry, "surf_pbx_entry.json")
        reloaded = loadfn("surf_pbx_entry.json")
        assert isinstance(reloaded.entry, ComputedEntry)
        assert reloaded.entry.data is not None

    def test_energy_functions(self):
        assert self.pbx_entry.energy_at_conditions(10, 0) == approx(35.93856, rel=1e-3), (
            "Wrong energy at pH = 10 and V = 0!"
        )
        assert self.pbx_entry.energy_at_conditions(np.array([1, 2, 3]), 0) == approx(
            [52.959, 51.068, 49.177], abs=1e-3
        ), "Wrong energy at pH = [1, 2, 3] and V = 0!"
        assert self.pbx_entry.energy_at_conditions(10, np.array([1, 2, 3])) == approx(
            [35.939, 35.939, 35.939], abs=1e-3
        ), "Wrong energy at pH = 10 and V = [1, 2, 3]!"
        assert self.pbx_entry.energy_at_conditions(
            np.array([1, 2, 3]), np.array([1, 2, 3])
        ) == approx(
            [52.959, 51.068, 49.177],
            abs=1e-3,
        ), "Wrong energy at pH = [1, 2, 3] and V = [1, 2, 3]!"


class TestPourbaixDiagram(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = loadfn(f"{TEST_DIR}/pourbaix_test_data.json")
        cls.pbx = PourbaixDiagram(cls.test_data["Zn"], filter_solids=True)
        cls.pbx_no_filter = PourbaixDiagram(cls.test_data["Zn"], filter_solids=False)

    def test_pourbaix_diagram(self):
        assert {entry.name for entry in self.pbx.stable_entries} == {
            "ZnO(s)",
            "Zn[2+]",
            "ZnHO2[-]",
            "ZnO2[2-]",
            "Zn(s)",
        }, "List of stable entries does not match"

        assert {entry.name for entry in self.pbx_no_filter.stable_entries} == {
            "Zn[2+]",
            "ZnHO2[-]",
            "ZnO2[2-]",
            "Zn(s)",
            "ZnO2(s)",
            "ZnO(s)",
        }, "List of stable entries for unfiltered pbx does not match"

        pbx_low_conc = PourbaixDiagram(
            self.test_data["Zn"], conc_dict={"Zn": 1e-8}, filter_solids=True
        )
        assert {entry.name for entry in pbx_low_conc.stable_entries} == {
            "Zn(HO)2(aq)",
            "Zn[2+]",
            "ZnHO2[-]",
            "ZnO2[2-]",
            "Zn(s)",
        }

    def test_properties(self):
        assert len(self.pbx.unstable_entries) == 2

    def test_multicomponent(self):
        # Assure no ions get filtered at high concentration
        ag_n = [entry for entry in self.test_data["Ag-Te-N"] if "Te" not in entry.composition]
        highconc = PourbaixDiagram(ag_n, filter_solids=True, conc_dict={"Ag": 1e-5, "N": 1})
        entry_sets = [set(entry.entry_id) for entry in highconc.stable_entries]
        assert {"mp-124", "ion-17"} in entry_sets

        # Binary system
        pd_binary = PourbaixDiagram(
            self.test_data["Ag-Te"],
            filter_solids=True,
            comp_dict={"Ag": 0.5, "Te": 0.5},
            conc_dict={"Ag": 1e-8, "Te": 1e-8},
        )
        assert len(pd_binary.stable_entries) == 30
        test_entry = pd_binary.find_stable_entry(8, 2)
        assert "mp-499" in test_entry.entry_id

        # Find a specific multi-entry to test
        assert pd_binary.get_decomposition_energy(test_entry, 8, 2) == 0

        pd_ternary = PourbaixDiagram(self.test_data["Ag-Te-N"], filter_solids=True)
        assert len(pd_ternary.stable_entries) == 49

        # Fetch a solid entry and a ground state entry mixture
        ag_te_n = self.test_data["Ag-Te-N"][-1]
        ground_state_ag_with_ions = MultiEntry(
            [self.test_data["Ag-Te-N"][i] for i in [4, 18, 30]],
            weights=[1 / 3, 1 / 3, 1 / 3],
        )
        assert pd_ternary.get_decomposition_energy(ag_te_n, 2, -1) == approx(2.767822855765)
        assert pd_ternary.get_decomposition_energy(ag_te_n, 10, -2) == approx(3.756840056890625)
        assert pd_ternary.get_decomposition_energy(ground_state_ag_with_ions, 2, -1) == approx(0)

        # Test invocation of Pourbaix diagram from ternary data
        new_ternary = PourbaixDiagram(pd_ternary.all_entries)
        assert len(new_ternary.stable_entries) == 49
        assert new_ternary.get_decomposition_energy(ag_te_n, 2, -1) == approx(2.767822855765)
        assert new_ternary.get_decomposition_energy(ag_te_n, 10, -2) == approx(3.756840056890625)
        assert new_ternary.get_decomposition_energy(ground_state_ag_with_ions, 2, -1) == approx(0)

        # Test processing of multi-entries with degenerate reaction, produced
        # a bug in a prior implementation
        entries = [
            PourbaixEntry(ComputedEntry("VFe2Si", -1.8542253150000008), entry_id="mp-4595"),
            PourbaixEntry(ComputedEntry("Fe", 0), entry_id="mp-13"),
            PourbaixEntry(ComputedEntry("V2Ir2", -2.141851640000006), entry_id="mp-569250"),
            PourbaixEntry(
                IonEntry(Ion.from_formula("Fe[2+]"), -0.7683100214319288), entry_id="ion-0"
            ),
            PourbaixEntry(
                IonEntry(Ion.from_formula("Li[1+]"), -3.0697590542787156), entry_id="ion-12"
            ),
        ]
        comp_dict = Composition({"Fe": 1, "Ir": 1, "Li": 2, "Si": 1, "V": 2}).fractional_composition

        multi_entry = PourbaixDiagram.process_multientry(entries, prod_comp=comp_dict)
        assert multi_entry is None

    def test_get_pourbaix_domains(self):
        domains = PourbaixDiagram.get_pourbaix_domains(self.test_data["Zn"])
        assert len(domains[0]) == 7

    def test_get_decomposition(self):
        # Test a stable entry to ensure that it's zero in the stable region
        entry = self.test_data["Zn"][12]  # Should correspond to mp-2133
        assert self.pbx.get_decomposition_energy(entry, 10, 1) == approx(0.0, 5), (
            "Decomposition energy of ZnO is not 0."
        )

        # Test an unstable entry to ensure that it's never zero
        entry = self.test_data["Zn"][11]
        ph, v = np.meshgrid(np.linspace(0, 14), np.linspace(-2, 4))
        result = self.pbx_no_filter.get_decomposition_energy(entry, ph, v)
        assert (result >= 0).all(), "Unstable energy has hull energy of 0 or less"

        # Test an unstable hydride to ensure HER correction works
        assert self.pbx.get_decomposition_energy(entry, -3, -2) == approx(3.6979147983333)
        # Test a list of pHs
        self.pbx.get_decomposition_energy(entry, np.linspace(0, 2, 5), 2)

        # Test a list of Vs
        self.pbx.get_decomposition_energy(entry, 4, np.linspace(-3, 3, 10))

        # Test a set of matching arrays
        ph, v = np.meshgrid(np.linspace(0, 14), np.linspace(-3, 3))
        self.pbx.get_decomposition_energy(entry, ph, v)

        # Test custom ions
        entries = self.test_data["C-Na-Sn"]
        ion = IonEntry(Ion.from_formula("NaO28H80Sn12C24+"), -161.676)
        custom_ion_entry = PourbaixEntry(ion, entry_id="some_ion")
        pbx = PourbaixDiagram(
            [*entries, custom_ion_entry],
            filter_solids=True,
            comp_dict={"Na": 1, "Sn": 12, "C": 24},
        )
        assert pbx.get_decomposition_energy(custom_ion_entry, 5, 2) == approx(
            2.1209002582, abs=1e-1
        )

    def test_get_stable_entry(self):
        entry = self.pbx.get_stable_entry(0, 0)
        assert entry.entry_id == "ion-0"

    def test_multielement_parallel(self):
        # Simple test to ensure that multiprocessing is working
        test_entries = self.test_data["Ag-Te-N"]
        nproc = multiprocessing.cpu_count()
        pbx = PourbaixDiagram(test_entries, filter_solids=True, nproc=nproc)
        assert len(pbx.stable_entries) == 49

    def test_solid_filter(self):
        entries = self.test_data["Zn"]
        pbx = PourbaixDiagram(entries, filter_solids=False)
        oxidized_phase = pbx.find_stable_entry(10, 2)
        assert oxidized_phase.name == "ZnO2(s)"

        entries = self.test_data["Zn"]
        pbx = PourbaixDiagram(entries, filter_solids=True)
        oxidized_phase = pbx.find_stable_entry(10, 2)
        assert oxidized_phase.name == "ZnO(s)"

    def test_serialization(self):
        dct = self.pbx.as_dict()
        new = PourbaixDiagram.from_dict(dct)
        assert {entry.name for entry in new.stable_entries} == {
            "ZnO(s)",
            "Zn[2+]",
            "ZnHO2[-]",
            "ZnO2[2-]",
            "Zn(s)",
        }, "List of stable entries does not match"

        # Test with unstable solid entries included (filter_solids=False), this should result in the
        # previously filtered entries being included
        dct = self.pbx_no_filter.as_dict()
        new = PourbaixDiagram.from_dict(dct)
        assert {entry.name for entry in new.stable_entries} == {
            "Zn[2+]",
            "ZnHO2[-]",
            "ZnO2[2-]",
            "Zn(s)",
            "ZnO2(s)",
            "ZnO(s)",
        }, "List of stable entries for unfiltered pbx does not match"

        pd_binary = PourbaixDiagram(
            self.test_data["Ag-Te"],
            filter_solids=True,
            comp_dict={"Ag": 0.5, "Te": 0.5},
            conc_dict={"Ag": 1e-8, "Te": 1e-8},
        )
        new_binary = PourbaixDiagram.from_dict(pd_binary.as_dict())
        assert len(pd_binary.stable_entries) == len(new_binary.stable_entries)

    def test_3D_pourbaix_domains(self):
        pH_limits = (-2, 16)
        phi_limits = (-2, 2)
        lg_conc_limits = (-12, -2)
        # Test that the Pourbaix diagram can be generated in 3D
        self.pbx._stable_3D_domains, self.pbx._stable_3D_domain_vertices = (
            self.pbx.get_3D_pourbaix_domains(
                self.pbx._processed_entries,
                limits=[pH_limits, phi_limits, lg_conc_limits],
            )
        )
        # TODO: test equilibrium 3D domains


class TestSurfacePourbaixDiagram(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_structure = Structure.from_file(f"{TEST_DIR}/SrIrO3_001_2x2x4.cif")
        cls.ref_entry = ComputedStructureEntry(cls.ref_structure, -133.3124)  # formation energy
        cls.test_data = loadfn(f"{TEST_DIR}/surface_pourbaix_test_data.json")
        cls.pbx = cls.test_data["reference_pourbaix_diagram"]
        pH_limits = [0, 4]
        phi_limits = [0, 1]  # limit to 2 domains for now
        cls.pbx.stable_entries, cls.pbx.stable_vertices = cls.pbx.get_pourbaix_domains(
            cls.pbx.all_entries, [pH_limits, phi_limits]
        )
        cls.surf_pbx = SurfacePourbaixDiagram(
            cls.test_data["surface_entries"],
            cls.ref_entry,
            cls.pbx,
            reference_surface_entry_factor=1.0,
        )
        cls.surf_pbx_wo_elems = SurfacePourbaixDiagram(
            cls.test_data["surface_entries"],
            cls.ref_entry,
            cls.pbx,
            reference_surface_entry_factor=1.0,
            reference_elements=("O", "Ir", "Sr"),
        )

    @staticmethod
    def domain_formulae(domain) -> list[str]:
        return [entry.composition.formula for entry in domain.entry_list]

    def test_ref_elems(self):
        # make sure it works for both specified and unspecified reference elements
        assert set(self.surf_pbx_wo_elems.ref_elems) == set(["O", "Ir", "Sr"]), (
            "Reference elements not inferred correctly"
        )
        assert set(self.surf_pbx.ref_elems) == set(["O", "Ir", "Sr"]), (
            "Reference elements not explicitly set correctly"
        )

    # TODO: make the tests more rigorous
    def test_ind_surface_pbx_entries(self):
        # Test that the surface pourbaix entries are correctly initialized
        assert len(self.surf_pbx.ind_surface_pbx_entries.values()) == 2

        for domain, surface_entries in self.surf_pbx.ind_surface_pbx_entries.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert len(surface_entries) == 3, "Incorrect number of surface entries"

            for entry in surface_entries:
                assert isinstance(entry, SurfacePourbaixEntry), (
                    "Entry is not a SurfacePourbaixEntry"
                )
                assert isinstance(entry.entry, ComputedEntry), "Entry is not a ComputedEntry"
                assert isinstance(entry.reference_entries, dict), (
                    "Reference entries are not a dictionary"
                )
                assert all(
                    isinstance(ref_entry, PourbaixEntry)
                    for ref_entry in entry.reference_entries.values()
                ), "Reference entries are not PourbaixEntries"
                assert len(entry.reference_entries) == 3, "Incorrect number of reference entries"
                assert set(entry.reference_entries.keys()) == set(["O", "Ir", "Sr"]), (
                    "Incorrect reference elements"
                )

                if set(TestSurfacePourbaixDiagram.domain_formulae(domain)) == set(["Sr1", "Ir1"]):
                    # Test SurfacePourbaixEntries are correct
                    if entry.composition.formula == "Sr16 Ir16 O48":
                        assert entry.npH == -96.0, "Wrong npH!"
                        assert entry.nPhi == -64.0, "Wrong nPhi!"
                        assert entry.nH2O == 0.0, "Wrong nH2O!"
                        assert entry.energy == approx(83.12870, rel=1e-3), "Wrong Energy!"

                    if entry.composition.formula == "Sr8 Ir16 O40":
                        assert entry.npH == -80.0, "Wrong npH!"
                        assert entry.nPhi == -64.0, "Wrong nPhi!"
                        assert entry.nH2O == 0.0, "Wrong nH2O!"
                        assert entry.energy == approx(67.43366, rel=1e-3), "Wrong Energy!"

                    if entry.composition.formula == "Ir16 O32":
                        assert entry.npH == -64.0, "Wrong npH!"
                        assert entry.nPhi == -64.0, "Wrong nPhi!"
                        assert entry.nH2O == 0.0, "Wrong nH2O!"
                        assert entry.energy == approx(48.71359, rel=1e-3), "Wrong Energy!"
                else:
                    if entry.composition.formula == "Sr16 Ir16 O48":
                        assert entry.npH == -32.0, "Wrong npH!"
                        assert entry.nPhi == 0.0, "Wrong nPhi!"
                        assert entry.nH2O == -32.0, "Wrong nH2O!"
                        assert entry.energy == approx(54.85056, rel=1e-3), "Wrong Energy!"

                    if entry.composition.formula == "Sr8 Ir16 O40":
                        assert entry.npH == -16.0, "Wrong npH!"
                        assert entry.nPhi == 0.0, "Wrong nPhi!"
                        assert entry.nH2O == -32.0, "Wrong nH2O!"
                        assert entry.energy == approx(39.15552, rel=1e-3), "Wrong Energy!"

                    if entry.composition.formula == "Ir16 O32":
                        assert entry.npH == 0.0, "Wrong npH!"
                        assert entry.nPhi == 0.0, "Wrong nPhi!"
                        assert entry.nH2O == -32.0, "Wrong nH2O!"
                        assert entry.energy == approx(20.43545, rel=1e-3), "Wrong Energy!"

    def test_ind_hyperplanes(self):
        # Test that the surface pourbaix hyperplanes are correctly initialized
        assert len(self.surf_pbx.ind_hyperplanes) == 2

        for domain, hyperplane_info in self.surf_pbx.ind_hyperplanes.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert hyperplane_info["hyperplanes"].shape == (8, 4), (
                "Incorrect dimensions of hyperplanes"
            )
            if set(TestSurfacePourbaixDiagram.domain_formulae(domain)) == set(["Sr1", "Ir1"]):
                # check each row is in the list, not necessarily in order
                for hyperplane in hyperplane_info["hyperplanes"]:
                    assert np.any(
                        np.allclose(hyperplane, row, rtol=1e-3, atol=1e-3)
                        for row in [
                            [5.6736, 64.0, 1.0, -83.1287],
                            [4.728, 64.0, 1.0, -67.43366],
                            [3.7824, 64.0, 1.0, -48.71359],
                            [0.059, 0.99826, 0.0, -0.44108],
                            [1.0, 0.0, 0.0, -4.0],
                            [-1.0, -0.0, 0.0, -0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, -268.20246],
                        ]
                    ), "Incorrect hyperplane"
                assert hyperplane_info["interior_point"] == approx(
                    [2.0, 0.16182, -134.52836], rel=1e-2
                ), "Incorrect interior point"
            else:
                # check each row is in the list, not necessarily in order
                for hyperplane in hyperplane_info["hyperplanes"]:
                    assert np.any(
                        np.allclose(hyperplane, row, rtol=1e-3, atol=1e-3)
                        for row in [
                            [1.8912, 0.0, 1.0, -54.85056],
                            [0.9456, 0.0, 1.0, -39.15552],
                            [0.0, 0.0, 1.0, -20.43545],
                            [-0.059, -0.99826, 0.0, 0.44108],
                            [1.0, -0.0, 0.0, -4.0],
                            [-1.0, 0.0, 0.0, -0.0],
                            [-0.0, 1.0, 0.0, -1.0],
                            [0.0, 0.0, -1.0, -124.83072],
                        ]
                    ), "Incorrect hyperplane"
                assert hyperplane_info["interior_point"] == approx(
                    [2.0, 0.66182, -62.41536], rel=1e-2
                ), "Incorrect interior point"

    def test_ind_stable_domain_vertices(self):
        # Test that the surface pourbaix domains are correctly initialized
        assert len(self.surf_pbx.ind_stable_domain_vertices) == 2

        for domain, stable_domains in self.surf_pbx.ind_stable_domain_vertices.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert len(stable_domains) == 1, "Incorrect number of stable domains"
            if set(TestSurfacePourbaixDiagram.domain_formulae(domain)) == set(["Sr1", "Ir1"]):
                for vertices in stable_domains.values():
                    assert np.allclose(
                        vertices,
                        [[0.0, 0.0], [4.0, 0.0], [4.0, 0.20545], [0.0, 0.44185]],
                        rtol=1e-3,
                        atol=1e-3,
                    )
            else:
                for vertices in stable_domains.values():
                    assert np.allclose(
                        vertices,
                        [[0.0, 0.44185], [4.0, 0.20545], [4.0, 1.0], [0.0, 1.0]],
                        rtol=1e-3,
                        atol=1e-3,
                    )

    def test_final_stable_domain_vertices(self):
        # Test that the final stable domain vertices are correctly initialized
        assert len(self.surf_pbx.stable_vertices) == 1

        # Only one domain in this case
        for domain, vertices in self.surf_pbx.stable_vertices.items():
            assert isinstance(domain, PourbaixEntry), "Domain is not a PourbaixEntry"
            # check each row is in the list, not necessarily in order
            for vertex in vertices:
                assert np.any(
                    np.allclose(vertex, row, rtol=1e-3, atol=1e-3)
                    for row in [
                        [0.0, 0.44185],
                        [4.0, 0.20545],
                        [4.0, -0.0],
                        [-0.0, 0.0],
                    ]
                ), "Incorrect vertex"

    def test_serialization(self):
        dct = self.surf_pbx.as_dict()
        new = SurfacePourbaixDiagram.from_dict(dct)
        # Test that the surface pourbaix entries are correctly initialized
        for domain, surface_entries in new.ind_surface_pbx_entries.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert len(surface_entries) == 3, "Incorrect number of surface entries"

            for entry in surface_entries:
                assert isinstance(entry, SurfacePourbaixEntry), (
                    "Entry is not a SurfacePourbaixEntry"
                )
                assert isinstance(entry.entry, ComputedEntry), "Entry is not a ComputedEntry"
                assert isinstance(entry.reference_entries, dict), (
                    "Reference entries are not a dictionary"
                )
                assert all(
                    isinstance(ref_entry, PourbaixEntry)
                    for ref_entry in entry.reference_entries.values()
                ), "Reference entries are not PourbaixEntries"
                assert len(entry.reference_entries) == 3, "Incorrect number of reference entries"
                assert set(entry.reference_entries.keys()) == set(["O", "Ir", "Sr"]), (
                    "Incorrect reference elements"
                )


class Test3DSurfacePourbaixDiagram(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = loadfn(f"{TEST_DIR}/surface_3d_pourbaix_test_data.json")
        cls.surf_form_entries = cls.test_data["surface_entries"]
        cls.ref_entry = cls.test_data["reference_entry"]
        cls.pbx = cls.test_data["reference_pourbaix_diagram"]
        cls.pH_limits = (0, 1)
        cls.phi_limits = (0, 1)
        cls.lg_conc_limits = (-3, -2)

        # We'll use the following later for 3D surface Pourbaix diagrams
        try:
            getattr(cls.pbx, "stable_3D_entries"), getattr(cls.pbx, "stable_3D_vertices")
        except AttributeError:
            cls.pbx._stable_3D_domains, cls.pbx._stable_3D_domain_vertices = (
                cls.pbx.get_3D_pourbaix_domains(
                    cls.pbx._processed_entries,
                    limits=[cls.pH_limits, cls.phi_limits, cls.lg_conc_limits],
                )
            )

        cls.surf_pbx = SurfacePourbaixDiagram(
            cls.surf_form_entries,
            cls.ref_entry,
            cls.pbx,
            reference_surface_entry_factor=0.25,
            process_3D=True,
            at_equilibrium=False,
        )

    @staticmethod
    def domain_formulae(domain) -> list[str]:
        return [entry.composition.formula for entry in domain.entry_list]

    def test_ref_elems(self):
        # make sure it works for both specified and unspecified reference elements
        assert set(self.surf_pbx.ref_elems) == set(["O", "La", "Mn", "H"]), (
            "Reference elements not inferred correctly"
        )

    def test_ind_3D_surface_pbx_entries(self):
        # Test that the surface pourbaix domains are correctly initialized
        assert len(self.surf_pbx.ind_3D_surface_pbx_entries.values()) == 1

        for domain, surface_entries in self.surf_pbx.ind_3D_surface_pbx_entries.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert len(surface_entries) == 3, "Incorrect number of surface entries"

            for entry in surface_entries:
                assert isinstance(entry, SurfacePourbaixEntry), (
                    "Entry is not a SurfacePourbaixEntry"
                )
                assert isinstance(entry.entry, ComputedEntry), "Entry is not a ComputedEntry"
                assert isinstance(entry.reference_entries, dict), (
                    "Reference entries are not a dictionary"
                )
                assert all(
                    isinstance(ref_entry, PourbaixEntry)
                    for ref_entry in entry.reference_entries.values()
                ), "Reference entries are not PourbaixEntries"
                assert len(entry.reference_entries) == 4, "Incorrect number of reference entries"
                assert set(entry.reference_entries.keys()) == set(["La", "Mn", "O", "H"]), (
                    "Incorrect reference elements"
                )

    def test_ind_3D_hyperplanes(self):
        # Test that the surface pourbaix hyperplanes are correctly initialized
        assert len(self.surf_pbx.ind_3D_surface_pbx_entries) == 1

        for domain, hyperplane_info in self.surf_pbx.ind_3D_hyperplanes.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert hyperplane_info["hyperplanes"].shape == (16, 5), (
                "Incorrect dimensions of hyperplanes"
            )
            if set(TestSurfacePourbaixDiagram.domain_formulae(domain)) == set(["La1", "Mn1"]):
                # check each row is in the list, not necessarily in order
                for hyperplane in hyperplane_info["hyperplanes"]:
                    assert np.any(
                        np.allclose(hyperplane, row, rtol=1e-3, atol=1e-3)
                        for row in [
                            [1.0638, 3.0, 0.3546, 1.0, -10.77591],
                            [0.97515, 3.0, 0.31028, 1.0, -11.41614],
                            [1.07858, 4.25, 0.32505, 1.0, -12.86195],
                            [0.0, -0.0, 1.0, 0.0, 2.0],
                            [-0.0, -1.0, -0.0, 0.0, -0.0],
                            [1.0, 0.0, -0.0, 0.0, -1.0],
                            [-0.0, -0.0, 1.0, 0.0, 2.0],
                            [-0.0, 1.0, 0.0, 0.0, -1.0],
                            [-0.0, -1.0, -0.0, 0.0, -0.0],
                            [-0.0, 0.0, -1.0, 0.0, -3.0],
                            [-1.0, 0.0, -0.0, 0.0, -0.0],
                            [-1.0, 0.0, -0.0, 0.0, -0.0],
                            [1.0, 0.0, -0.0, 0.0, -1.0],
                            [0.0, 1.0, -0.0, 0.0, -1.0],
                            [-0.0, 0.0, -1.0, 0.0, -3.0],
                            [0.0, 0.0, 0.0, -1.0, -38.50865],
                        ]
                    ), "Incorrect hyperplane"
                assert hyperplane_info["interior_point"] == approx(
                    [0.5, 0.5, -2.5, -19.25433], rel=1e-2
                ), "Incorrect interior point"

    def test_ind_3D_stable_domain_vertices(self):
        # Test that the surface pourbaix domains are correctly initialized
        assert len(self.surf_pbx.ind_3D_surface_pbx_entries) == 1

        for domain, stable_domains in self.surf_pbx.ind_3D_stable_domain_vertices.items():
            assert isinstance(domain, MultiEntry), "Domain is not a Pourbaix MultiEntry"
            assert len(stable_domains) == 1, "Incorrect number of stable domains"
            if set(TestSurfacePourbaixDiagram.domain_formulae(domain)) == set(["La1", "Mn1"]):
                for vertices in stable_domains.values():
                    assert np.allclose(
                        vertices,
                        [
                            [-0.0, 0.0, -3.0],
                            [-0.0, -0.0, -2.0],
                            [1.0, -0.0, -2.0],
                            [1.0, -0.0, -3.0],
                            [1.0, 1.0, -3.0],
                            [1.0, 1.0, -2.0],
                            [0.0, 1.0, -3.0],
                            [-0.0, 1.0, -2.0],
                        ],
                        rtol=1e-3,
                        atol=1e-3,
                    )

    def test_final_stable_domain_vertices(self):
        # Test that the final stable domain vertices are correctly initialized
        assert len(self.surf_pbx._stable_3D_domain_vertices) == 1

        # Only one domain in this case
        for domain, vertices in self.surf_pbx._stable_3D_domain_vertices.items():
            assert isinstance(domain, PourbaixEntry), "Domain is not a PourbaixEntry"
            # check each row is in the list, not necessarily in order
            for vertex in vertices:
                assert np.any(
                    np.allclose(vertex, row, rtol=1e-3, atol=1e-3)
                    for row in [
                        [0.0, 0.0, -2.0],
                        [1.0, 0.0, -3.0],
                        [1.0, 0.0, -2.0],
                        [1.0, 1.0, -3.0],
                        [1.0, 1.0, -2.0],
                        [0.0, 1.0, -3.0],
                        [0.0, 1.0, -2.0],
                    ]
                ), "Incorrect vertex"


class Test3DEquilibriumSurfacePourbaixDiagram(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = loadfn(f"{TEST_DIR}/surface_3d_eq_pourbaix_test_data.json")
        cls.surf_form_entries = cls.test_data["surface_entries"]
        cls.ref_entry = cls.test_data["reference_entry"]
        cls.pbx = cls.test_data["reference_pourbaix_diagram"]
        cls.pH_limits = (10, 14)
        cls.phi_limits = (-0.5, 0.5)
        cls.lg_conc_limits = (-8, -3)

    def test_at_equilibrium_3D_pourbaix_domains(self):
        ref_LMO_entry = [
            entry for entry in self.pbx._processed_entries if entry.name == "LaMnO3(s)"
        ][0]

        # Get at-equilibrium 3D bulk Pourbaix domains
        guess_interior_point = [12, 0.0, -6, -5.4894]
        self.pbx.stable_3D_entries, self.pbx.stable_3D_vertices = self.pbx.get_3D_pourbaix_domains(
            self.pbx._processed_entries,
            limits=[self.pH_limits, self.phi_limits, self.lg_conc_limits],
            at_equilibrium=True,
            ref_pbx_entry=ref_LMO_entry,
            interior_point=guess_interior_point,
        )

        assert len(self.pbx.stable_3D_vertices) == 8, (
            "Number of stable 3D vertices is not correct. "
            f"Expected 8, got {len(self.pbx.stable_3D_vertices)}"
        )

        surf_pbx = SurfacePourbaixDiagram(
            self.surf_form_entries,
            self.ref_entry,
            self.pbx,
            reference_surface_entry_factor=0.25,
            process_3D=True,
            at_equilibrium=True,
            excluded_bulk_entries=[ref_LMO_entry],
        )
        pourbaix_domains_3D, pourbaix_domain_vertices_3D = (
            surf_pbx._stable_3D_domains,
            surf_pbx._stable_3D_domain_vertices,
        )
        assert len(pourbaix_domains_3D) == 1, (
            "Number of stable 3D domains is not correct. "
            f"Expected 1, got {len(pourbaix_domains_3D)}"
        )
        assert len(pourbaix_domain_vertices_3D) == 1, (
            "Number of stable 3D vertices is not correct. "
            f"Expected 1, got {len(pourbaix_domain_vertices_3D)}"
        )


class TestPourbaixPlotter(TestCase):
    def setUp(self):
        self.test_data = loadfn(f"{TEST_DIR}/pourbaix_test_data.json")
        self.pd = PourbaixDiagram(self.test_data["Zn"])
        self.plotter = PourbaixPlotter(self.pd)

    def test_plot_pourbaix(self):
        plotter = PourbaixPlotter(self.pd)
        # Default limits
        plotter.get_pourbaix_plot()
        # Non-standard limits
        plotter.get_pourbaix_plot(limits=[[-5, 4], [-2, 2]])

    def test_plot_entry_stability(self):
        entry = self.pd.all_entries[0]
        self.plotter.plot_entry_stability(entry, limits=[[-2, 14], [-3, 3]])

        # binary system
        pd_binary = PourbaixDiagram(self.test_data["Ag-Te"], comp_dict={"Ag": 0.5, "Te": 0.5})
        binary_plotter = PourbaixPlotter(pd_binary)
        ax = binary_plotter.plot_entry_stability(self.test_data["Ag-Te"][53])
        assert isinstance(ax, plt.Axes)


class TestSurfacePourbaixPlotter(TestCase):
    def setUp(self):
        self.ref_structure = Structure.from_file(f"{TEST_DIR}/SrIrO3_001_2x2x4.cif")
        self.ref_entry = ComputedStructureEntry(self.ref_structure, -133.3124)  # formation energy
        self.test_data = loadfn(f"{TEST_DIR}/surface_pourbaix_test_data.json")
        self.pbx = self.test_data["reference_pourbaix_diagram"]
        self.pH_limits = [0, 4]
        self.phi_limits = [0, 1]  # limit to 2 domains for now
        self.pbx.stable_entries, self.pbx.stable_vertices = self.pbx.get_pourbaix_domains(
            self.pbx.all_entries, [self.pH_limits, self.phi_limits]
        )
        self.test_data["surface_entries"][0].entry_id = "A_0"
        self.surf_pbx = SurfacePourbaixDiagram(
            self.test_data["surface_entries"],
            self.ref_entry,
            self.pbx,
            reference_surface_entry_factor=1.0,
        )
        self.plotter = PourbaixPlotter(self.surf_pbx)

    def test_plot_surface_pourbaix(self):
        plotter = PourbaixPlotter(self.surf_pbx)
        # Default limits
        plotter.get_pourbaix_plot()
        # Non-standard limits
        plotter.get_pourbaix_plot(limits=[self.pH_limits, self.phi_limits])

    def test_get_energy_vs_potential_plot(self):
        pH = 0
        # Test that the energy vs potential plot is generated correctly
        ax = self.plotter.get_energy_vs_potential_plot(
            pH,
            energy_range=(-3, 3),
            reference_entry_id="A_0",
            V_range=self.phi_limits,
            full_formula=True,
        )
        assert isinstance(ax, plt.Axes), "Plot is not an Axes object"
