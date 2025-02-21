"""
This module is intended to be used to compute Pourbaix diagrams of arbitrary compositions
and formation energies.
"""

from __future__ import annotations

import itertools
import logging
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb

from pymatgen.analysis.chempot_diagram import get_centroid_2d, simple_pca
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify

if TYPE_CHECKING:
    from typing import Any

    import matplotlib.pyplot as plt
    from typing_extensions import Self

__author__ = "Sai Jayaraman"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.4"
__maintainer__ = "Joseph Montoya"
__credits__ = "Arunima Singh, Joseph Montoya, Anjli Patel"
__email__ = "joseph.montoya@tri.global"
__status__ = "Production"
__date__ = "Nov 1, 2012"

# If you use this module in your work, consider citing:
due.cite(
    Doi("10.1103/PhysRevB.85.235438"),
    description="Prediction of solid-aqueous equilibria: Scheme to combine first-principles"
    "calculations of solids with experimental aqueous states",
)
due.cite(
    Doi("10.1021/acs.chemmater.7b03980"),
    description="Electrochemical Stability of Metastable Materials",
)
due.cite(
    Doi("10.1039/C9CP04799A"),
    description="Efficient Pourbaix diagrams of many-element compounds",
)

logger = logging.getLogger(__name__)

PREFAC = 0.0591
ELEMENTS_HO = {Element("H"), Element("O")}
SYMBOLS_HO = {elt.symbol for elt in ELEMENTS_HO}


# TODO: Revise to more closely reflect PDEntry, invoke from energy/composition
# TODO: PourbaixEntries depend implicitly on having entry energies be
# formation energies, should be a better way to get from raw energies
# TODO: uncorrected_energy is a bit of a misnomer, but not sure what to rename
class PourbaixEntry(MSONable, Stringify):
    """
    An object encompassing all data relevant to a solid or ion
    in a Pourbaix diagram. Each bulk solid/ion has an energy
    g of the form: e = e0 + 0.0591 log10(conc) - nO mu_H2O
    + (nH - 2nO) pH + phi (nH - 2nO - q).

    Note that the energies corresponding to the input entries
    should be formation energies with respect to hydrogen and
    oxygen gas in order for the Pourbaix diagram formalism to
    work. This may be changed to be more flexible in the future.
    """

    def __init__(self, entry, entry_id=None, concentration=1e-6):
        """
        Args:
            entry (ComputedEntry/ComputedStructureEntry/PDEntry/IonEntry): An
                entry object
            entry_id ():
            concentration ():
        """
        self.entry = entry
        if isinstance(entry, IonEntry):
            self.concentration = concentration
            self.phase_type = "Ion"
            self.charge = entry.ion.charge
        else:
            self.concentration = 1.0
            self.phase_type = "Solid"
            self.charge = 0
        self.uncorrected_energy = entry.energy
        if entry_id is not None:
            self.entry_id = entry_id
        elif hasattr(entry, "entry_id") and entry.entry_id:
            self.entry_id = entry.entry_id
        else:
            self.entry_id = None

    @property
    def npH(self):
        """The number of H."""
        return self.entry.composition.get("H", 0) - 2 * self.entry.composition.get("O", 0)

    @property
    def nH2O(self):
        """The number of H2O."""
        return self.entry.composition.get("O", 0)

    @property
    def nPhi(self):
        """The number of electrons."""
        return self.npH - self.charge

    @property
    def n_conc(self):
        """The conc number used for 3D plots that vary concentration. 1 for ions, 0 for solids."""
        return int(isinstance(self.entry, IonEntry))

    # TODO: write tests for n_conc

    @property
    def energy_without_conc_term(self):
        """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
        # Note: this implicitly depends on formation energies as input
        return self.uncorrected_energy - (MU_H2O * self.nH2O)

    @property
    def energy(self):
        """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
        # Note: this implicitly depends on formation energies as input
        return self.uncorrected_energy - (MU_H2O * self.nH2O) + self.conc_term

    @property
    def energy_per_atom(self):
        """Energy per atom of the Pourbaix entry."""
        return self.energy / self.composition.num_atoms

    def __getattr__(self, attr):
        """Get attributed normalized by number of non H or O atoms, e.g.
        for Zn2O6, energy / 2 or for AgTe3(OH)3, energy / 4."""
        # supported normalized attributes
        if attr in (
            "normalized_npH",
            "normalized_nPhi",
            "normalized_nH2O",
            "normalized_n_conc",
            "normalized_energy",
            "normalized_energy_without_conc_term",
        ):
            attr = attr.replace("normalized_", "")
            return self.__getattribute__(attr) * self.normalization_factor

        # default other attributes
        return self.__getattribute__(attr)

    @property
    def name(self):
        """The entry's name."""
        if self.phase_type == "Solid":
            return f"{self.entry.reduced_formula}(s)"

        return self.entry.name

    @property
    def elements(self):
        """Elements in the entry."""
        return self.entry.elements

    def energy_at_conditions(self, pH, V):
        """Get free energy for a given pH and V.

        Args:
            pH (float): pH at which to evaluate free energy
            V (float): voltage at which to evaluate free energy

        Returns:
            free energy at conditions
        """
        return self.energy + self.npH * PREFAC * pH + self.nPhi * V

    def get_element_fraction(self, element):
        """Get the elemental fraction of a given non-OH element.

        Args:
            element (Element or str): string or element corresponding
                to element to get from composition

        Returns:
            fraction of element / sum(all non-OH elements)
        """
        return self.composition.get(element) * self.normalization_factor

    def normalized_energy_at_conditions(self, pH, V):
        """Energy at an electrochemical condition, compatible with
        numpy arrays for pH/V input.

        Args:
            pH (float): pH at condition
            V (float): applied potential at condition

        Returns:
            energy normalized by number of non-O/H atoms at condition
        """
        return self.energy_at_conditions(pH, V) * self.normalization_factor

    @property
    def conc_term(self):
        """The concentration contribution to the free energy. Should only be present
        when there are ions in the entry.
        """
        return PREFAC * np.log10(self.concentration)

    # TODO: not sure if these are strictly necessary with refactor
    def as_dict(self):
        """Get dict which contains Pourbaix Entry data.
        Note that the pH, voltage, H2O factors are always calculated when
        constructing a PourbaixEntry object.
        """
        dct = {"@module": type(self).__module__, "@class": type(self).__name__}
        if isinstance(self.entry, IonEntry):
            dct["entry_type"] = "Ion"
        else:
            dct["entry_type"] = "Solid"
        dct["entry"] = self.entry.as_dict()
        dct["concentration"] = self.concentration
        dct["entry_id"] = self.entry_id
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Invokes a PourbaixEntry from a dictionary."""
        entry_type = dct["entry_type"]
        entry = (
            IonEntry.from_dict(dct["entry"])
            if entry_type == "Ion"
            else MontyDecoder().process_decoded(dct["entry"])
        )
        entry_id = dct["entry_id"]
        concentration = dct["concentration"]
        return cls(entry, entry_id, concentration)

    @property
    def normalization_factor(self):
        """Sum of number of atoms minus the number of H and O in composition."""
        return 1.0 / (self.num_atoms - self.composition.get("H", 0) - self.composition.get("O", 0))

    @property
    def composition(self):
        """Composition."""
        return self.entry.composition

    @property
    def num_atoms(self):
        """Number of atoms in current formula. Useful for normalization."""
        return self.composition.num_atoms

    def to_pretty_string(self, full_formula=False) -> str:
        """A pretty string representation."""
        if self.entry_id:
            return f"{self.entry_id}"
        if self.phase_type == "Solid":
            if full_formula:
                return f"{self.entry.formula}(s)"  # return full formula if requested
            return f"{self.entry.reduced_formula}(s)"

        return self.entry.name

    def __repr__(self):
        energy, npH, nPhi, nH2O, entry_id = (
            self.energy,
            self.npH,
            self.nPhi,
            self.nH2O,
            self.entry_id,
        )
        return f"{type(self).__name__}({self.entry.composition} with {energy=:.4f}, {npH=}, {nPhi=}, {nH2O=}, {entry_id=})"


class OxygenPourbaixEntry(PourbaixEntry):
    """Pourbaix entry for oxygen. This is a special case because we count the number of O
    in composition for the normalized energy and disregard it for npH and nH2O."""

    def __init__(
        self,
        entry: ComputedEntry,
        entry_id: Optional[str] = None,
        concentration: float = 1.0,
    ):
        super().__init__(entry, entry_id, concentration)
        self.phase_type = "Liquid"

    @property
    def npH(self):
        """The number of H."""
        return self.entry.composition.get("H", 0)

    @property
    def nH2O(self):
        """The number of H2O."""
        return 0  # all O atoms are counted as O

    @property
    def normalization_factor(self):
        """Sum of number of atoms minus the number of H in composition."""
        return 1.0 / (self.num_atoms - self.composition.get("H", 0))

    @property
    def name(self):
        """The entry's name."""
        if self.phase_type == "Liquid":
            return f"{self.entry.reduced_formula}(l)"

        return self.entry.name

    def as_dict(self):
        """Get dict which contains Oxygen Pourbaix Entry data."""
        dct = {"@module": type(self).__module__, "@class": type(self).__name__}
        dct["entry_type"] = "Liquid"
        dct["entry"] = self.entry.as_dict()
        dct["concentration"] = self.concentration
        dct["entry_id"] = self.entry_id
        return dct

    def to_pretty_string(self) -> str:
        """A pretty string representation."""
        return self.name


class HydrogenPourbaixEntry(PourbaixEntry):
    """Pourbaix entry for hydrogen. This is a special case because we disregard the ion concentration
    which is assumed to be H+. Concentration of H+ is already factored into the pH term.
    """

    def __init__(
        self,
        entry: ComputedEntry,
        entry_id: str | None = None,
        concentration: float = 1.0,
    ):
        super().__init__(entry, entry_id, concentration)
        self.phase_type = "Ion"  # concentration is the same as pH, so we neglect it

    @property
    def npH(self):
        """The number of H."""
        return -self.entry.composition.get("H", 0) - 2 * self.entry.composition.get("O", 0)

    @property
    def nPhi(self):
        """The number of electrons."""
        return self.npH

    @property
    def n_conc(self):
        """The conc number used for 3D plots that vary concentration. Does not apply here."""
        return 0

    # use the PourbaixEntry energy property since we've defined concentration as 1.0
    # @property
    # def energy(self):
    #     """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
    #     # Note: this implicitly depends on formation energies as input
    #     return self.uncorrected_energy - (MU_H2O * self.nH2O)

    @property
    def normalization_factor(self):
        """Sum of number of atoms minus the number of H and O in composition."""
        return 1.0 / (self.num_atoms - 3 * self.composition.get("O", 0))  # last term assumes H2O


class SurfacePourbaixEntry(PourbaixEntry):
    """
    Energy of each entry is the surface free energy determined with respect to the reference species
    (most stable species in the Pourbaix diagram) at each pH and V. The surface energy is
    calculated using a two-step surface-solvent dissociation-solvation process.

    Step 1: Surface is dissociated into constituent atoms in their standard states.
    Step 2: The atoms are solvated in the solvent to form the most stable species.

    Surface free energy is calculated as -(E_step_1 + E_step_2), where -E_step_1 can be interpreted
    as the formation energy. E_step_2 is the solvation energy of the atoms in the solvent.

    Also borrowing from surface_analysis.SlabEntry
    """

    def __init__(
        self,
        surface_entry: ComputedStructureEntry,
        reference_entries: dict[str, PourbaixEntry],
        clean_entry: Optional[ComputedStructureEntry] = None,
        clean_entry_factor: float = 1.0,
        entry_id: Optional[float] = None,
        concentration: float = 1e-6,
        label: Optional[str] = None,
        marker: Optional[str] = None,
        color: Optional[str] = None,
    ):
        """
        Args:
            surface_entry (ComputedStructureEntry): Surface entry
            reference_entries (dict[str, PourbaixEntry]): Reference entries
            clean_entry (ComputedStructureEntry): Pristine surface entry
            clean_entry_factor (float): Factor to scale the clean entry, can be used to account for supercell size
            entry_id (float): Entry ID
            concentration (float): Concentration of surface entry species, does not apply to solids
                TODO: might want to get rid of concentration
            label (str): Label for plotting
            marker (str): Marker for plotting
            color (str): Color for plotting
        """
        super().__init__(surface_entry, entry_id, concentration)
        self.surface_entry = surface_entry  # energy must be formation energy
        self.reference_entries = reference_entries
        if not all(isinstance(entry, PourbaixEntry) for entry in self.reference_entries.values()):
            raise ValueError("All reference entries must be PourbaixEntry objects")
        if not any(entry.name == "H2O(l)" for entry in self.reference_entries.values()):
            raise ValueError("H2O must be included in reference entries")
        self.label = label
        self.clean_entry = clean_entry
        self.clean_entry_factor = clean_entry_factor
        self.mark = marker
        self.color = color

    @property
    def npH(self):
        """The number of H by counting the number of atoms over reference entries"""
        return -sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_npH
            for element in self.entry.composition.elements
        )

    @property
    def nPhi(self):
        """The number of electrons by counting the number of atoms over reference entries"""
        return -sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_nPhi
            for element in self.entry.composition.elements
        )

    @property
    def nH2O(self):
        """The number of H2O by counting the number of atoms over reference entries"""
        return -sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_nH2O
            for element in self.entry.composition.elements
        )

    @property
    def n_conc(self):
        """The conc number used for 3D plots that vary concentration. Depends on the reference entries."""
        return -sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_n_conc
            for element in self.entry.composition.elements
        )

    @property
    def energy_delta_G1(self):
        """The energy in the first step of the dissociation-solvation reaction. Expressed as the
        negative of the formation energy of the surface entry."""
        return -self.entry.energy

    @property
    def energy_delta_G2(self):
        """Total free energy change for the second step of the dissociation-solvation reaction."""
        return sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_energy
            for element in self.entry.composition.elements
        )

    @property
    def energy_delta_G2_without_conc_term(self):
        """Total free energy change for the second step of the dissociation-solvation reaction without the
        concentration term."""
        return sum(
            self.entry.composition.get(element)
            * self.reference_entries[element.symbol].normalized_energy_without_conc_term
            for element in self.entry.composition.elements
        )

    @property
    def energy(self):
        """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
        return -(self.energy_delta_G1 + self.energy_delta_G2)

    @property
    def energy_without_conc_term(self):
        """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
        return -(self.energy_delta_G1 + self.energy_delta_G2_without_conc_term)

    @property
    def energy_per_atom(self):
        """Energy per atom of the Pourbaix entry."""
        return self.energy / self.composition.num_atoms

    @property
    def elements(self):
        """Elements in the entry."""
        return self.entry.elements

    @property
    def name(self):
        """The entry's name."""
        if self.phase_type == "Solid":
            return f"{self.entry.formula}(s)"

        return self.entry.name

    def as_dict(self):
        """Get dict which contains Surface Pourbaix Entry data.
        Note that the pH, voltage, H2O factors are always calculated when
        constructing a SurfacePourbaixEntry object.
        """
        dct = {"@module": type(self).__module__, "@class": type(self).__name__}
        if isinstance(self.entry, IonEntry):
            dct["entry_type"] = "Ion"
        else:
            dct["entry_type"] = "Solid"
        dct["entry"] = self.entry.as_dict()
        dct["reference_entries"] = {k: v.as_dict() for k, v in self.reference_entries.items()}
        dct["concentration"] = self.concentration
        dct["entry_id"] = self.entry_id
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Invokes a SurfacePourbaixEntry from a dictionary."""
        entry_type = dct["entry_type"]
        surface_entry = (
            IonEntry.from_dict(dct["entry"])
            if entry_type == "Ion"
            else MontyDecoder().process_decoded(dct["entry"])
        )
        reference_entries = {
            k: MontyDecoder().process_decoded(v) for k, v in dct["reference_entries"].items()
        }
        entry_id = dct["entry_id"]
        concentration = dct["concentration"]
        return cls(surface_entry, reference_entries, entry_id, concentration)

    @property
    def get_unit_primitive_area(self):
        """The surface area of the adsorbed system per unit area of the primitive slab system."""
        curr_surface_area = self.surface_area(self.entry)
        pristine_surface_area = self.surface_area(self.clean_entry) * self.clean_entry_factor
        return curr_surface_area / pristine_surface_area

    @classmethod
    def surface_area(cls, structure):
        """The surface area of the adsorbed system."""
        while hasattr(structure, "structure"):
            structure = structure.structure
        try:
            matrix = structure.lattice.matrix
        except AttributeError as e:
            raise ValueError(
                "Structure must be a ComputedStructureEntry or similar and have a lattice.matrix"
            ) from e

        return np.linalg.norm(np.cross(matrix[0], matrix[1]))

    @property
    def normalization_factor(self):
        """Using unit primitive area as normalization factor."""
        return 1.0 / self.get_unit_primitive_area


class MultiEntry(PourbaixEntry):
    """
    PourbaixEntry-like object for constructing multi-elemental Pourbaix diagrams.
    """

    def __init__(self, entry_list, weights=None):
        """Initialize a MultiEntry.

        Args:
            entry_list ([PourbaixEntry]): List of component PourbaixEntries
            weights ([float]): Weights associated with each entry. Default is None
        """
        self.weights = weights or [1.0] * len(entry_list)
        self.entry_list = entry_list

    def __getattr__(self, attr):
        """
        Because most of the attributes here are just weighted averages of the entry_list,
        we save some space by having a set of conditionals to define the attributes.
        """
        # Attributes that are weighted averages of entry attributes
        if attr in [
            "energy",
            "npH",
            "nH2O",
            "nPhi",
            "n_conc",
            "conc_term",
            "composition",
            "uncorrected_energy",
            "energy_without_conc_term",
            "elements",
        ]:
            # TODO: Composition could be changed for compat with sum
            start = Composition() if attr == "composition" else 0
            weighted_values = (
                getattr(entry, attr) * weight
                for entry, weight in zip(self.entry_list, self.weights)
            )
            return sum(weighted_values, start)

        # Attributes that are just lists of entry attributes
        if attr in ["entry_id", "phase_type"]:
            return [getattr(entry, attr) for entry in self.entry_list]

        # normalization_factor, num_atoms should work from superclass
        return self.__getattribute__(attr)

    @property
    def name(self):
        """MultiEntry name, i. e. the name of each entry joined by ' + '."""
        return " + ".join(entry.name for entry in self.entry_list)

    def __repr__(self):
        energy, npH, nPhi, nH2O, entry_id = (
            self.energy,
            self.npH,
            self.nPhi,
            self.nH2O,
            self.entry_id,
        )
        cls_name, species = type(self).__name__, self.name
        return (
            f"Pourbaix{cls_name}({energy=:.4f}, {npH=}, {nPhi=}, {nH2O=}, {entry_id=}, {species=})"
        )

    def as_dict(self):
        """Get MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "entry_list": [entry.as_dict() for entry in self.entry_list],
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            MultiEntry
        """
        entry_list = [PourbaixEntry.from_dict(entry) for entry in dct.get("entry_list", ())]
        return cls(entry_list, dct.get("weights"))


# TODO: this class isn't particularly useful in its current form, could be
# refactored to include information about the reference solid
class IonEntry(PDEntry):
    """
    Object similar to PDEntry, but contains an Ion object instead of a
    Composition object.

    Attributes:
        name (str): A name for the entry. This is the string shown in the phase diagrams.
            By default, this is the reduced formula for the composition, but can be
            set to some other string for display purposes.
    """

    def __init__(self, ion: Ion, energy: float, name: str | None = None, attribute=None):
        """
        Args:
            ion: Ion object
            energy: Energy for composition.
            name: Optional parameter to name the entry. Defaults to the
                chemical formula.
            attribute: Optional attribute of the entry, e.g. band gap.
        """
        self.ion = ion
        # Auto-assign name
        name = name or self.ion.reduced_formula
        super().__init__(composition=ion.composition, energy=energy, name=name, attribute=attribute)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Get an IonEntry object from a dict."""
        return cls(
            Ion.from_dict(dct["ion"]),
            dct["energy"],
            dct.get("name"),
            dct.get("attribute"),
        )

    def as_dict(self):
        """Create a dict of composition, energy, ion name, and attribute."""
        return {
            "ion": self.ion.as_dict(),
            "energy": self.energy,
            "name": self.name,
            "attribute": self.attribute,
        }

    def __repr__(self):
        return f"IonEntry : {self.composition} with energy = {self.energy:.4f}"

    @property
    def formula(self) -> str:
        """A formula string with appended charge. The
        charge is written with the sign preceding the magnitude, e.g.
        'Ca1 +2'. Uncharged species have "(aq)" appended, e.g. "O2 (aq)".
        """
        return self.ion.formula

    @property
    def reduced_formula(self) -> str:
        """A reduced formula string with appended charge. The
        charge is placed in brackets with the sign preceding the magnitude, e.g.
        'Ca[+2]'. Uncharged species have "(aq)" appended, e.g. "O2(aq)".
        """
        return self.ion.reduced_formula


def ion_or_solid_comp_object(formula):
    """Get an Ion or Composition object given a formula.

    Args:
        formula: String formula. Eg. of ion: NaOH(aq), Na[+];
            Eg. of solid: Fe2O3(s), Fe(s), Na2O

    Returns:
        Composition/Ion object
    """
    if re.match(r"\[([^\[\]]+)\]|\(aq\)", formula):
        comp_obj = Ion.from_formula(formula)
    elif re.search(r"\(s\)", formula):
        comp_obj = Composition(formula[:-3])
    else:
        comp_obj = Composition(formula)
    return comp_obj


# TODO: the solids filter breaks some of the functionality of the
# heatmap plotter, because the reference states for decomposition
# don't include oxygen/hydrogen in the OER/HER regions


# TODO: create a from_phase_diagram class method for non-formation energy invocation
# TODO: invocation from a MultiEntry entry list could be a bit more robust
# TODO: serialization is still a bit rough around the edges
class PourbaixDiagram(MSONable):
    """Create a Pourbaix diagram from entries."""

    def __init__(
        self,
        entries: list[PourbaixEntry] | list[MultiEntry],
        comp_dict: dict[str, float] | None = None,
        conc_dict: dict[str, float] | None = None,
        filter_solids: bool = True,
        nproc: int | None = None,
        pH_limits: tuple[float, float] = (-2, 16),
        phi_limits: tuple[float, float] = (-2, 2),
        lg_conc_limits: tuple[float, float] = (-12, -2),
        process_3D: bool = False,
    ):
        """
        Args:
            entries ([PourbaixEntry] or [MultiEntry]): Entries list
                containing Solids and Ions or a list of MultiEntries
            comp_dict (dict[str, float]): Dictionary of compositions,
                defaults to equal parts of each elements
            conc_dict (dict[str, float]): Dictionary of ion concentrations,
                defaults to 1e-6 for each element
            filter_solids (bool): applying this filter to a Pourbaix
                diagram ensures all included solid phases are filtered by
                stability on the compositional phase diagram. Defaults to True.
                The practical consequence of this is that highly oxidized or reduced
                phases that might show up in experiments due to kinetic limitations
                on oxygen/hydrogen evolution won't appear in the diagram, but they are
                not actually "stable" (and are frequently overstabilized from DFT errors).
                Hence, including only the stable solid phases generally leads to the
                most accurate Pourbaix diagrams.
            nproc (int): number of processes to generate multi-entries with
                in parallel. Defaults to None (serial processing).
            pH_limits (tuple): pH limits for Pourbaix diagram. Defaults to (-2, 16).
            phi_limits (tuple): Potential limits for Pourbaix diagram. Defaults to (-2, 2).
            lg_conc_limits (tuple): Log concentration limits for 3D Pourbaix diagram. Defaults to (-12, -2).
            process_3D (bool): Whether to process the Pourbaix diagram in 3D. Defaults to False.
        """
        entries = deepcopy(entries)
        self.filter_solids = filter_solids

        # Get non-OH elements
        self.pbx_elts = list(
            set(itertools.chain.from_iterable([entry.composition.elements for entry in entries]))
            - ELEMENTS_HO
        )
        self.dim = len(self.pbx_elts) - 1

        # Process multi-entry inputs
        if isinstance(entries[0], MultiEntry):
            self._processed_entries = entries
            # Extract individual entries
            single_entries = list(
                set(itertools.chain.from_iterable([entry.entry_list for entry in entries]))
            )
            self._unprocessed_entries = single_entries
            self._filtered_entries = single_entries
            self._conc_dict = None
            self._elt_comp = {
                k: v for k, v in entries[0].composition.items() if k not in ELEMENTS_HO
            }
            self._multi_element = True

        # Process single entry inputs
        else:
            # Set default conc/comp dicts
            if not comp_dict:
                comp_dict = {elt.symbol: 1 / len(self.pbx_elts) for elt in self.pbx_elts}
            if not conc_dict:
                conc_dict = {elt.symbol: 1e-6 for elt in self.pbx_elts}
            self._conc_dict = conc_dict

            self._elt_comp = comp_dict
            self.pourbaix_elements = self.pbx_elts

            solid_entries = [entry for entry in entries if entry.phase_type == "Solid"]
            ion_entries = [entry for entry in entries if entry.phase_type == "Ion"]

            # If a conc_dict is specified, override individual entry concentrations
            for entry in ion_entries:
                ion_elts = list(set(entry.elements) - ELEMENTS_HO)
                # TODO: the logic here for ion concentration setting is in two
                # places, in PourbaixEntry and here, should be consolidated
                if len(ion_elts) == 1:
                    entry.concentration = conc_dict[ion_elts[0].symbol] * entry.normalization_factor
                elif len(ion_elts) > 1 and not entry.concentration:
                    raise ValueError(
                        "Elemental concentration not compatible with multi-element ions"
                    )

            self._unprocessed_entries = solid_entries + ion_entries

            if len(solid_entries + ion_entries) != len(entries):
                raise ValueError(
                    'All supplied entries must have a phase type of either "Solid" or "Ion"'
                )

            if self.filter_solids:
                # O is 2.46 b/c pbx entry finds energies referenced to H2O
                entries_HO = [ComputedEntry("H", 0), ComputedEntry("O", 2.46)]
                solid_pd = PhaseDiagram(solid_entries + entries_HO)
                solid_entries = list(set(solid_pd.stable_entries) - set(entries_HO))

            self._filtered_entries = solid_entries + ion_entries
            if len(comp_dict) > 1:
                self._multi_element = True
                self._processed_entries = self._preprocess_pourbaix_entries(
                    self._filtered_entries, nproc=nproc
                )
            else:
                self._processed_entries = self._filtered_entries
                self._multi_element = False

        # Sort entries by name
        self._processed_entries = sorted(self._processed_entries, key=lambda x: x.name)

        self._stable_domains, self._stable_domain_vertices = self.get_pourbaix_domains(
            self._processed_entries, limits=[pH_limits, phi_limits]
        )

        if process_3D:
            self._stable_3D_domains, self._stable_3D_domain_vertices = self.get_3D_pourbaix_domains(
                self._processed_entries,
                limits=[pH_limits, phi_limits, lg_conc_limits],
            )

    def _convert_entries_to_points(self, pourbaix_entries):
        """
        Args:
            pourbaix_entries ([PourbaixEntry]): list of Pourbaix entries
                to process into vectors in nph-nphi-composition space.

        Returns:
            list of vectors, [[nph, nphi, e0, x1, x2, ..., xn-1]]
            corresponding to each entry in nph-nphi-composition space
        """
        vecs = [
            [entry.npH, entry.nPhi, entry.energy]
            + [entry.composition.get(elt) for elt in self.pbx_elts[:-1]]
            for entry in pourbaix_entries
        ]
        vecs = np.array(vecs)
        norms = np.transpose([[entry.normalization_factor for entry in pourbaix_entries]])
        vecs *= norms
        return vecs

    def _get_hull_in_nph_nphi_space(self, entries) -> tuple[list[PourbaixEntry], list[Simplex]]:
        """Generate convex hull of Pourbaix diagram entries in composition,
        npH, and nphi space. This enables filtering of multi-entries
        such that only compositionally stable combinations of entries
        are included.

        Args:
            entries ([PourbaixEntry]): list of PourbaixEntries to construct
                the convex hull

        Returns:
            tuple[list[PourbaixEntry], list[Simplex]]: PourbaixEntry list and stable
                facets corresponding to that list
        """
        ion_entries = [entry for entry in entries if entry.phase_type == "Ion"]
        solid_entries = [entry for entry in entries if entry.phase_type == "Solid"]

        # Pre-filter solids based on min at each composition
        logger.debug("Pre-filtering solids by min energy at each composition")
        sorted_entries = sorted(
            solid_entries,
            key=lambda x: (x.composition.reduced_composition, x.entry.energy_per_atom),
        )
        grouped_by_composition = itertools.groupby(
            sorted_entries, key=lambda x: x.composition.reduced_composition
        )
        min_entries = [
            next(iter(grouped_entries)) for comp, grouped_entries in grouped_by_composition
        ]
        min_entries += ion_entries

        logger.debug("Constructing nph-nphi-composition points for qhull")

        vecs = self._convert_entries_to_points(min_entries)
        maxes = np.max(vecs[:, :3], axis=0)
        extra_point = np.concatenate([maxes, np.ones(self.dim) / self.dim], axis=0)

        # Add padding for extra point
        pad = 1000
        extra_point[2] += pad
        points = np.concatenate([vecs, np.array([extra_point])], axis=0)
        logger.debug("Constructing convex hull in nph-nphi-composition space")
        hull = ConvexHull(points, qhull_options="QJ i")

        # Create facets and remove top
        facets = [facet for facet in hull.simplices if len(points) - 1 not in facet]

        if self.dim > 1:
            logger.debug("Filtering facets by Pourbaix composition")
            valid_facets = []
            for facet in facets:
                comps = vecs[facet][:, 3:]
                full_comps = np.concatenate(
                    [comps, 1 - np.sum(comps, axis=1).reshape(len(comps), 1)], axis=1
                )
                # Ensure an compositional interior point exists in the simplex
                if np.linalg.matrix_rank(full_comps) > self.dim:
                    valid_facets.append(facet)
        else:
            valid_facets = facets

        return min_entries, valid_facets

    def _preprocess_pourbaix_entries(self, entries, nproc=None):
        """Generate multi-entries for Pourbaix diagram.

        Args:
            entries ([PourbaixEntry]): list of PourbaixEntries to preprocess
                into MultiEntries
            nproc (int): number of processes to be used in parallel
                treatment of entry combos

        Returns:
            list[MultiEntry]: stable MultiEntry candidates
        """
        # Get composition
        tot_comp = Composition(self._elt_comp)
        # TODO: Assume this generation with fixed conc is valid for
        # 3D Pourbaix diagram that varies aqueous species conentration
        min_entries, valid_facets = self._get_hull_in_nph_nphi_space(entries)

        combos = []
        for facet in valid_facets:
            for idx in range(1, self.dim + 2):
                these_combos = []
                for combo in itertools.combinations(facet, idx):
                    these_entries = [min_entries[i] for i in combo]
                    these_combos.append(frozenset(these_entries))
                combos.append(these_combos)

        all_combos = set(itertools.chain.from_iterable(combos))

        list_combos = []
        for idx in all_combos:
            list_combos.append(list(idx))
        all_combos = list_combos

        multi_entries = []

        # Parallel processing of multi-entry generation
        if nproc is not None:
            func = partial(self.process_multientry, prod_comp=tot_comp)
            with Pool(nproc) as proc_pool:
                multi_entries = list(proc_pool.imap(func, all_combos))
            multi_entries = list(filter(bool, multi_entries))
        else:
            # Serial processing of multi-entry generation
            for combo in all_combos:
                multi_entry = self.process_multientry(combo, prod_comp=tot_comp)
                if multi_entry:
                    multi_entries.append(multi_entry)

        return multi_entries

    def _generate_multielement_entries(self, entries, nproc=None):
        """
        Create entries for multi-element Pourbaix construction.

        This works by finding all possible linear combinations
        of entries that can result in the specified composition
        from the initialized comp_dict.

        Args:
            entries ([PourbaixEntries]): list of Pourbaix entries
                to process into MultiEntries
            nproc (int): number of processes to be used in parallel
                treatment of entry combos
        """
        n_elems = len(self._elt_comp)  # No. of elements
        total_comp = Composition(self._elt_comp)

        # generate all combinations of compounds that have all elements
        entry_combos = [itertools.combinations(entries, idx + 1) for idx in range(n_elems)]
        entry_combos = itertools.chain.from_iterable(entry_combos)

        entry_combos = filter(lambda x: total_comp < MultiEntry(x).composition, entry_combos)

        # Generate and filter entries
        processed_entries = []
        total = sum(comb(len(entries), idx + 1) for idx in range(n_elems))
        if total > 1e6:
            warnings.warn(
                f"Your Pourbaix diagram includes {total} entries and may take a long time to generate."
            )

        # Parallel processing of multi-entry generation
        if nproc is not None:
            func = partial(self.process_multientry, prod_comp=total_comp)
            with Pool(nproc) as proc_pool:
                processed_entries = list(proc_pool.imap(func, entry_combos))
            processed_entries = list(filter(bool, processed_entries))
        # Serial processing of multi-entry generation
        else:
            for entry_combo in entry_combos:
                processed_entry = self.process_multientry(entry_combo, total_comp)
                if processed_entry is not None:
                    processed_entries.append(processed_entry)

        return processed_entries

    @staticmethod
    def process_multientry(entry_list, prod_comp, coeff_threshold=1e-4):
        """Static method for finding a multientry based on
        a list of entries and a product composition.
        Essentially checks to see if a valid aqueous
        reaction exists between the entries and the
        product composition and returns a MultiEntry
        with weights according to the coefficients if so.

        Args:
            entry_list ([Entry]): list of entries from which to
                create a MultiEntry
            prod_comp (Composition): composition constraint for setting
                weights of MultiEntry
            coeff_threshold (float): threshold of stoichiometric
                coefficients to filter, if weights are lower than
                this value, the entry is not returned
        """
        dummy_oh = [Composition("H"), Composition("O")]
        try:
            # Get balanced reaction coeffs, ensuring all < 0 or conc thresh
            # Note that we get reduced compositions for solids and non-reduced
            # compositions for ions because ions aren't normalized due to
            # their charge state.
            # Sort entries by name
            entry_list = sorted(entry_list, key=lambda x: x.name)

            entry_comps = [entry.composition for entry in entry_list]
            rxn = Reaction(entry_comps + dummy_oh, [prod_comp])
            react_coeffs = [-coeff for coeff in rxn.coeffs[: len(entry_list)]]
            all_coeffs = [*react_coeffs, rxn.get_coeff(prod_comp)]

            # Check if reaction coeff threshold met for Pourbaix compounds
            # All reactant/product coefficients must be positive nonzero
            if all(coeff > coeff_threshold for coeff in all_coeffs):
                return MultiEntry(entry_list, weights=react_coeffs)

            return None
        except ReactionError:
            return None

    @staticmethod
    def get_pourbaix_domains(pourbaix_entries, limits=None):
        """Get a set of Pourbaix stable domains (i. e. polygons) in
        pH-V space from a list of pourbaix_entries.

        This function works by using scipy's HalfspaceIntersection
        function to construct all of the 2-D polygons that form the
        boundaries of the planes corresponding to individual entry
        gibbs free energies as a function of pH and V. Hyperplanes
        of the form a*pH + b*V + 1 - g(0, 0) are constructed and
        supplied to HalfspaceIntersection, which then finds the
        boundaries of each Pourbaix region using the intersection
        points.

        Args:
            pourbaix_entries ([PourbaixEntry]): Pourbaix entries
                with which to construct stable Pourbaix domains
            limits ([[float]]): limits in which to do the pourbaix
                analysis

        Returns:
            Returns a dict of the form {entry: [boundary_points]}.
            The list of boundary points are the sides of the N-1
            dim polytope bounding the allowable ph-V range of each entry.
        """
        # Sort entries by name
        pourbaix_entries = sorted(pourbaix_entries, key=lambda x: x.name)

        if limits is None:
            limits = [[-2, 16], [-4, 4]]

        # Get hyperplanes
        hyperplanes = [
            np.array([-PREFAC * entry.npH, -entry.nPhi, 0, -entry.energy])
            * entry.normalization_factor
            for entry in pourbaix_entries
        ]
        hyperplanes = np.array(hyperplanes)
        hyperplanes[:, 2] = 1

        g_max = PourbaixDiagram.get_min_energy(limits, hyperplanes)

        # Add border hyperplanes and generate HalfspaceIntersection
        border_hyperplanes = [
            [-1, 0, 0, limits[0][0]],
            [1, 0, 0, -limits[0][1]],
            [0, -1, 0, limits[1][0]],
            [0, 1, 0, -limits[1][1]],
            [0, 0, -1, 2 * g_max],
        ]
        hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
        interior_point = [*np.mean(limits, axis=1).tolist(), g_max]
        hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

        # organize the boundary points by entry
        pourbaix_domains = {entry: [] for entry in pourbaix_entries}
        for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
            for v in facet:
                if v < len(pourbaix_entries):
                    this_entry = pourbaix_entries[v]
                    pourbaix_domains[this_entry].append(intersection)

        # Remove entries with no Pourbaix region
        pourbaix_domains = {k: v for k, v in pourbaix_domains.items() if v}
        pourbaix_domain_vertices = {}

        for entry, points in pourbaix_domains.items():
            points = np.array(points)[:, :2]
            # Initial sort to ensure consistency
            points = points[np.lexsort(np.transpose(points))]
            center = np.mean(points, axis=0)
            points_centered = points - center

            # Sort points by cross product of centered points,
            # isn't strictly necessary but useful for plotting tools
            points_centered = sorted(
                points_centered, key=cmp_to_key(lambda x, y: x[0] * y[1] - x[1] * y[0])
            )
            points = points_centered + center

            # Create simplices corresponding to Pourbaix boundary
            simplices = [Simplex(points[indices]) for indices in ConvexHull(points).simplices]
            pourbaix_domains[entry] = simplices
            pourbaix_domain_vertices[entry] = points
        return pourbaix_domains, pourbaix_domain_vertices

    @staticmethod
    def get_3D_pourbaix_domains(
        pourbaix_entries: list[PourbaixEntry],
        limits: list[list[float]] | None = None,
        at_equilibrium: bool = False,
        ref_pbx_entry: PourbaixEntry | None = None,
        interior_point: list[float] | None = None,
    ) -> dict[PourbaixEntry, list[list[float]]]:
        """Get a set of Pourbaix stable domains (i. e. polygons) in
        pH-V-conc space from a list of pourbaix_entries.

        This function works by using scipy's HalfspaceIntersection
        function to construct all of the 3-D polygons that form the
        boundaries of the hyperplanes corresponding to individual entry
        gibbs free energies as a function of pH, V, and conc. Hyperplanes
        of the form a*pH + b*V + c*log(conc) + 1 - g(0, 0) are constructed and
        supplied to HalfspaceIntersection, which then finds the
        boundaries of each Pourbaix region using the intersection
        points.

        New energy_at_conditions(pH, V, conc) should be
        self.energy_without_conc + self.npH * PREFAC * pH + self.nPhi * V + PREFAC * np.log10(conc).

        If at_equilibrium is True, the function will add an additional hyperplane corresponding to the
        equilibrium condition of the reference entry. The ref_pbx_entry is assumed to be already in the
        list of pourbaix_entries. The new hyperplane will point in the opposite direction compared with
        the other hyperplanes, effectively constraining the Pourbaix diagram to be one dimension lower
        than the non-equilibrium case.

        Args:
            pourbaix_entries ([PourbaixEntry]): Pourbaix entries
                with which to construct stable Pourbaix domains
            limits ([[float]]): limits in which to do the pourbaix
                analysis
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium
                conditions or not
            ref_pbx_entry (PourbaixEntry): entry to use as a reference for
                eqilibrium calculations
            interior_point ([float]): guess for the interior point of the
                HalfspaceIntersection object

        Returns:
            Returns a dict of the form {entry: [boundary_points]}.
            The list of boundary points are the sides of the N-1
            dim polytope bounding the allowable ph-V-log(conc) range of each entry.
        """

        # Sort entries by name
        pourbaix_entries = sorted(pourbaix_entries, key=lambda x: x.name)

        # Limits correspond to pH, V, and log(conc)
        # log10(conc) in practice should be -5 or lower
        if limits is None:
            limits = [[-2, 16], [-4, 4], [-12, -2]]

        # Get hyperplanes
        hyperplanes = [
            np.array(
                [
                    -PREFAC * entry.npH,
                    -entry.nPhi,
                    -PREFAC * entry.n_conc,
                    0,
                    -entry.energy_without_conc_term,
                ]
            )
            * entry.normalization_factor
            for entry in pourbaix_entries
        ]
        hyperplanes = np.array(hyperplanes)
        hyperplanes[:, -2] = 1
        # Add additional hyperplane to constrain the Pourbaix diagram at equilibrium
        if at_equilibrium:
            if ref_pbx_entry is None:
                raise ValueError(
                    "Must provide a reference Pourbaix entry for equilibrium calculations."
                )
            ref_pbx_hyperplane = (
                -np.array(
                    [
                        -PREFAC * ref_pbx_entry.npH,
                        -ref_pbx_entry.nPhi,
                        -PREFAC * ref_pbx_entry.n_conc,
                        0,
                        -(
                            ref_pbx_entry.energy_without_conc_term
                            - 1e-4  # can't go thicker than this
                        ),  # add a small offset to avoid numerical issues
                    ]
                )
                * ref_pbx_entry.normalization_factor
            )
            ref_pbx_hyperplane[-2] = -1
            hyperplanes = np.vstack([hyperplanes, ref_pbx_hyperplane])
        g_max = PourbaixDiagram.get_min_energy(limits, hyperplanes)

        # Add border hyperplanes and generate HalfspaceIntersection
        border_hyperplanes = [
            [-1, 0, 0, 0, limits[0][0]],
            [1, 0, 0, 0, -limits[0][1]],
            [0, -1, 0, 0, limits[1][0]],
            [0, 1, 0, 0, -limits[1][1]],
            [0, 0, -1, 0, limits[2][0]],
            [0, 0, 1, 0, -limits[2][1]],
            [0, 0, 0, -1, 2 * g_max],
        ]
        hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])
        # Interior point needs to be inside hyperplane of any reference entry
        # Strongly recommend to guess an interior_point for at_equilibrium calculations
        if interior_point is None:
            interior_point = [*np.mean(limits, axis=1).tolist(), g_max]

        hs_int = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

        # organize the boundary points by entry
        pourbaix_domains = {entry: [] for entry in pourbaix_entries}
        for intersection, facet in zip(hs_int.intersections, hs_int.dual_facets):
            for v in facet:
                if v < len(pourbaix_entries):
                    this_entry = pourbaix_entries[v]
                    pourbaix_domains[this_entry].append(intersection)

        # Remove entries with no Pourbaix region
        pourbaix_domains = {k: v for k, v in pourbaix_domains.items() if v}
        pourbaix_domain_vertices = {}

        for entry, points in pourbaix_domains.items():
            points_3d = np.array(points)[:, :3]
            # Initial sort to ensure consistency
            points_3d = points_3d[np.lexsort(np.transpose(points_3d))]

            points_2d, _v, w = simple_pca(points_3d, k=2)
            domain = ConvexHull(points_2d)
            centroid_2d = get_centroid_2d(points_2d[domain.vertices])
            ann_loc = centroid_2d @ w.T + np.mean(points_3d.T, axis=1)

            if at_equilibrium:
                simplices = [Simplex(points_3d[indices]) for indices in domain.simplices]
                vertices = points_3d[domain.vertices]  # ordered in 2D space
            else:
                simplices = [
                    Simplex(points_3d[indices]) for indices in ConvexHull(points_3d).simplices
                ]
                vertices = points_3d

            # center = np.mean(points, axis=0)
            # points_centered = points - center

            # TODO: Can probably do some sorting for annotations
            # Sort points by cross product of centered points,
            # isn't strictly necessary but useful for plotting tools
            # points_centered = sorted(points_centered, key=cmp_to_key(lambda x, y: x[0] * y[1] - x[1] * y[0]))
            # points = points_centered + center

            # Create simplices corresponding to Pourbaix boundary
            pourbaix_domains[entry] = simplices
            pourbaix_domain_vertices[entry] = vertices

        return pourbaix_domains, pourbaix_domain_vertices

    @staticmethod
    def get_min_energy(limits, hyperplanes):
        """Get the lower bound energy for the Pourbaix diagram.

        Args:
            limits ([[float]]): limits of pH and V for the Pourbaix diagram
            hyperplanes (np.array): hyperplanes corresponding to
                the Pourbaix entries

        Returns:
            float: lower bound energy for the Pourbaix diagram
        """
        max_contribs = np.max(np.abs(hyperplanes), axis=0)
        return np.dot(-max_contribs, [*np.max(np.abs(limits), axis=1).tolist(), 0, 1])

    def find_stable_entry(self, pH, V):
        """Find stable entry at a pH,V condition

        Args:
            pH (float): pH to find stable entry
            V (float): V to find stable entry.

        Returns:
            PourbaixEntry: stable entry at pH, V
        """
        return self.get_stable_entry(pH, V)  # alias

    def get_decomposition_energy(self, entry, pH, V):
        """Find decomposition to most stable entries in eV/atom,
        supports vectorized inputs for pH and V.

        Args:
            entry (PourbaixEntry): PourbaixEntry corresponding to
                compound to find the decomposition for
            pH (float, [float]): pH at which to find the decomposition
            V (float, [float]): voltage at which to find the decomposition

        Returns:
            Decomposition energy for the entry, i. e. the energy above
                the "Pourbaix hull" in eV/atom at the given conditions
        """
        # Check composition consistency between entry and Pourbaix diagram:
        pbx_comp = Composition(self._elt_comp).fractional_composition
        entry_pbx_comp = Composition(
            {elt: coeff for elt, coeff in entry.composition.items() if elt not in ELEMENTS_HO}
        ).fractional_composition
        if entry_pbx_comp != pbx_comp:
            raise ValueError("Composition of stability entry does not match Pourbaix Diagram")
        entry_normalized_energy = entry.normalized_energy_at_conditions(pH, V)
        hull_energy = self.get_hull_energy(pH, V)
        decomposition_energy = entry_normalized_energy - hull_energy

        # Convert to eV/atom instead of eV/normalized formula unit
        decomposition_energy /= entry.normalization_factor
        decomposition_energy /= entry.composition.num_atoms
        return decomposition_energy

    def get_hull_energy(self, pH, V):
        """Get the minimum energy of the Pourbaix "basin" that is formed
        from the stable Pourbaix planes. Vectorized.

        Args:
            pH (float or [float]): pH at which to find the hull energy
            V (float or [float]): V at which to find the hull energy

        Returns:
            np.array: minimum Pourbaix energy at conditions
        """
        all_gs = np.array(
            [entry.normalized_energy_at_conditions(pH, V) for entry in self.stable_entries]
        )
        return np.min(all_gs, axis=0)

    def get_stable_entry(self, pH, V):
        """Get the stable entry at a given pH, V condition.

        Args:
            pH (float): pH at a given condition
            V (float): V at a given condition

        Returns:
            PourbaixEntry | MultiEntry: Pourbaix or multi-entry
                corresponding to the minimum energy entry at a given pH, V condition
        """
        all_gs = np.array(
            [entry.normalized_energy_at_conditions(pH, V) for entry in self.stable_entries]
        )
        return self.stable_entries[np.argmin(all_gs)]

    @property
    def stable_entries(self):
        """The stable entries in the Pourbaix diagram."""
        return list(self._stable_domains)

    @stable_entries.setter
    def stable_entries(self, entries):
        """Set the stable entries in the Pourbaix diagram."""
        self._stable_domains = entries

    @property
    def stable_3D_entries(self):
        """The stable entries in the pH-V-log(conc) 3D Pourbaix diagram."""
        return list(self._stable_3D_domains)

    @stable_3D_entries.setter
    def stable_3D_entries(self, entries):
        """Set the stable entries in the pH-V-log(conc) 3D Pourbaix diagram."""
        self._stable_3D_domains = entries

    @property
    def unstable_entries(self):
        """All unstable entries in the Pourbaix diagram."""
        return [entry for entry in self.all_entries if entry not in self.stable_entries]

    @property
    def stable_vertices(self):
        """Vertices corresponding to each stable Pourbaix entry."""
        return self._stable_domain_vertices

    @stable_vertices.setter
    def stable_vertices(self, vertices):
        """Set the stable vertices in the Pourbaix diagram."""
        self._stable_domain_vertices = vertices

    @property
    def stable_3D_vertices(self):
        """3D vertices in pH-V-log(conc) corresponding to each stable Pourbaix entry."""
        return self._stable_3D_domain_vertices

    @stable_3D_vertices.setter
    def stable_3D_vertices(self, vertices):
        """Set the stable pH-V-log(conc) 3D vertices in the Pourbaix diagram."""
        self._stable_3D_domain_vertices = vertices

    @property
    def all_entries(self):
        """All entries used to generate the Pourbaix diagram."""
        return self._processed_entries

    @property
    def unprocessed_entries(self):
        """Unprocessed entries."""
        return self._unprocessed_entries

    def as_dict(self):
        """Get MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "entries": [entry.as_dict() for entry in self._unprocessed_entries],
            "comp_dict": self._elt_comp,
            "conc_dict": self._conc_dict,
            "filter_solids": self.filter_solids,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            PourbaixDiagram
        """
        decoded_entries = MontyDecoder().process_decoded(dct["entries"])
        return cls(
            decoded_entries,
            comp_dict=dct.get("comp_dict"),
            conc_dict=dct.get("conc_dict"),
            filter_solids=bool(dct.get("filter_solids")),
        )


class SurfacePourbaixDiagram(MSONable):
    """Create a Pourbaix diagram from surface energies. Energies are calculated as the formation energy from stable
    species at each pH and V.

    Refer to SurfacePourbaixEntry for more information on the construction of surface Pourbaix entries.
    """

    def __init__(
        self,
        surface_entries: Iterable[ComputedEntry],
        reference_surface_entry: ComputedEntry,
        reference_pourbaix_diagram: PourbaixDiagram,
        reference_surface_entry_factor: float = 1.0,
        reference_elements: Optional[Iterable[str]] = None,
        excluded_bulk_entries: Optional[Iterable[PourbaixEntry]] = None,
        process_3D: bool = False,
        at_equilibrium: bool = False,
    ) -> None:
        """
        Args:
            surface_entries: list of ComputedEntry's with surface energies
            reference_surface_entry: ComputedEntry for the reference surface
            reference_pourbaix_diagram: PourbaixDiagram for the bulk phase
            reference_surface_entry_factor: factor to scale the reference surface entry energy
            reference_elements: elements to be considered as reference in the surface Pourbaix diagram
            excluded_bulk_entries: list of bulk Pourbaix domains to exclude from the surface Pourbaix diagram
            process_3D: whether to process the 3D Pourbaix diagram
        """
        self.surface_entries = surface_entries  # with surface formation energies
        self.reference_surface_entry = reference_surface_entry
        self.reference_surface_entry_factor = reference_surface_entry_factor
        self.ref_pbx = reference_pourbaix_diagram
        self.ref_elems = reference_elements or self.get_ref_elems()

        # Create 3D Pourbaix diagram in pH-V-log(conc) space
        if process_3D:
            # Assumes we have already constructed the 3D bulk Pourbaix diagram
            try:
                (
                    getattr(self.ref_pbx, "stable_3D_entries"),
                    getattr(self.ref_pbx, "stable_3D_vertices"),
                )
            except AttributeError:
                raise AttributeError(
                    "3D Pourbaix diagram not found. Please run get_3D_pourbaix_domains() on the reference Pourbaix diagram and set stable_3D_entries and stable_3D_vertices."
                )
                # self.ref_pbx.stable_3D_entries, self.ref_pbx.stable_3D_vertices = self.ref_pbx.get_3D_pourbaix_domains(
                #     self.ref_pbx.all_entries
                # )
            # Sort the bulk Pourbaix entries first
            self.ref_pbx.stable_3D_vertices = dict(
                sorted(self.ref_pbx.stable_3D_vertices.items(), key=lambda x: x[0].name)
            )

            # Remove any excluded bulk entries
            if excluded_bulk_entries is not None:
                for entry in excluded_bulk_entries:
                    self.ref_pbx._stable_3D_domains.pop(entry, None)
                    self.ref_pbx._stable_3D_domain_vertices.pop(entry, None)

            # Step 1: create SurfacePourbaixEntry's for each region
            self.ind_3D_surface_pbx_entries = self.construct_surf_pbx_entries(process_3D=True)

            # Step 2: construct individual hyperplanes for each region
            self.ind_3D_hyperplanes = self.construct_hyperplanes(
                process_3D=True, at_equilibrium=at_equilibrium
            )  # at eq not useful

            # Step 3: construct Pourbaix domains for each region
            self.ind_3D_stable_domains, self.ind_3D_stable_domain_vertices = (
                self.construct_pourbaix_domains(process_3D=True, at_equilibrium=at_equilibrium)
            )  # at eq not useful

            # Step 4: construct overall Pourbaix domains
            # styling compatible with PourbaixDiagram class
            self._stable_3D_domains, self._stable_3D_domain_vertices = self.merge_pourbaix_domains(
                process_3D=True, at_equilibrium=at_equilibrium
            )

        # Create regular Pourbaix diagram in pH-V space
        else:
            # Sort the bulk Pourbaix entries first
            self.ref_pbx.stable_vertices = dict(
                sorted(self.ref_pbx.stable_vertices.items(), key=lambda x: x[0].name)
            )

            # Step 1: create SurfacePourbaixEntry's for each region
            self.ind_surface_pbx_entries = self.construct_surf_pbx_entries()

            # Step 2: construct individual hyperplanes for each region
            self.ind_hyperplanes = self.construct_hyperplanes()

            # Step 3: construct Pourbaix domains for each region
            self.ind_stable_domains, self.ind_stable_domain_vertices = (
                self.construct_pourbaix_domains()
            )

            # Step 4: construct overall Pourbaix domains
            # styling compatible with PourbaixDiagram class
            self._stable_domains, self._stable_domain_vertices = self.merge_pourbaix_domains()

    @property
    def H2O_pourbaix_entry(self) -> OxygenPourbaixEntry:
        """Pourbaix entry for H2O.

        Returns:
            Pourbaix entry for H2O.
        """
        return OxygenPourbaixEntry(ComputedEntry("H8O4", -9.83319943))

    @property
    def H_ion_pourbaix_entry(self) -> IonEntry:
        """Pourbaix entry for H+.

        Returns:
            Pourbaix entry for H+.
        """
        return HydrogenPourbaixEntry(IonEntry(Ion.from_formula("H[1+]"), 0.0))

    @property
    def stable_entries(self) -> list[PourbaixEntry]:
        """The stable entries in the surface Pourbaix diagram.

        Returns:
            list of Pourbaix entries each corresponding to a stable domain.
        """
        return list(self._stable_domains)

    @stable_entries.setter
    def stable_entries(self, entries: dict[PourbaixEntry, Simplex]) -> None:
        """Set the stable entries in the surface Pourbaix diagram.

        Args:
            entries: dictionary of stable Pourbaix entries and their corresponding simplices.
        """
        self._stable_domains = entries

    @property
    def stable_3D_entries(self):
        """The stable entries in the pH-V-log(conc) 3D surface Pourbaix diagram."""
        return list(self._stable_3D_domains)

    @stable_3D_entries.setter
    def stable_3D_entries(self, entries):
        """Set the stable entries in the pH-V-log(conc) 3D surface Pourbaix diagram."""
        self._stable_3D_domains = entries

    @property
    def stable_vertices(self) -> dict[PourbaixEntry, np.ndarray]:
        """Vertices corresponding to each stable Pourbaix entry.

        Returns:
            dictionary of stable Pourbaix entries and their corresponding vertices.
        """
        return self._stable_domain_vertices

    @stable_vertices.setter
    def stable_vertices(self, vertices: dict[PourbaixEntry, np.ndarray]) -> None:
        """Set the dictionary of stable vertices in the surface Pourbaix diagram.

        Args:
            vertices: dictionary of stable Pourbaix entries and their corresponding vertices.
        """
        self._stable_domain_vertices = vertices

    @property
    def stable_3D_vertices(self):
        """3D vertices in pH-V-log(conc) corresponding to each stable surface Pourbaix entry."""
        return self._stable_3D_domain_vertices

    @stable_3D_vertices.setter
    def stable_3D_vertices(self, vertices):
        """Set the stable pH-V-log(conc) 3D vertices in the surface Pourbaix diagram."""
        self._stable_3D_domain_vertices = vertices

    def get_ref_elems(self) -> tuple[str]:
        """Get reference elements for the surface Pourbaix diagram.

        Returns:
            tuple of reference elements in str.
        """
        ref_elems = set(
            itertools.chain.from_iterable(
                [entry.composition.elements for entry in self.surface_entries]
            )
        )
        return tuple(elem.symbol for elem in ref_elems)

    def _construct_surf_pbx_entries_for_domain(
        self, domain: PourbaixEntry
    ) -> list[SurfacePourbaixEntry]:
        """Construct the surface Pourbaix entries for each domain.

        Args:
            domain: Pourbaix entry or MultiEntry for which to construct the surface Pourbaix entries.

        Returns:
            list of SurfacePourbaixEntry for the domain.
        """
        if isinstance(domain, MultiEntry):
            entries = domain.entry_list
        elif isinstance(domain, PourbaixEntry):
            entries = [domain]
        else:
            raise ValueError("Invalid entry type")

        # match entries to reference elements based on composition
        ref_comps = sorted(set(self.ref_elems) - SYMBOLS_HO)
        entry_comps = sorted(
            entries,  # there should be only 1 non-OH element per entry
            key=lambda entry: list(set(entry.composition.elements) - ELEMENTS_HO).pop().symbol,
        )
        assert len(ref_comps) == len(entry_comps), (
            "Mismatch in number of reference and entry compositions"
        )
        ref_entry_map = dict(zip(ref_comps, entry_comps))

        # Use special OxygenPourbaixEntry for 1/2 O2(g) + 2H+(aq) + 2e- -> H2O(l)
        if "O" in self.ref_elems:
            ref_entry_map["O"] = self.H2O_pourbaix_entry

        # Use special HydrogenPourbaixEntry for 1/2 H2(g) -> H+(aq) + e-
        if "H" in self.ref_elems:
            ref_entry_map["H"] = self.H_ion_pourbaix_entry

        return [
            SurfacePourbaixEntry(
                surf_entry,
                ref_entry_map,
                clean_entry=self.reference_surface_entry,
                clean_entry_factor=self.reference_surface_entry_factor,
                entry_id=surf_entry.entry_id,
            )
            for surf_entry in self.surface_entries
        ]

    # Both 2D and 3D surface Pourbaix diagram case
    def construct_surf_pbx_entries(
        self, process_3D: bool = False
    ) -> dict[PourbaixEntry, SurfacePourbaixEntry]:
        """Construct the surface Pourbaix entries for each stable domain in the original Pourbaix diagram.

        Args:
            process_3D: whether we are processing a 3D Pourbaix diagram

        Returns:
            dictionary of surface Pourbaix entries for each stable domain.
        """
        pbx_entries = {}
        stable_vertices = (
            self.ref_pbx.stable_3D_vertices if process_3D else self.ref_pbx.stable_vertices
        )
        for entry in stable_vertices:
            surf_pbx_entries = self._construct_surf_pbx_entries_for_domain(entry)
            pbx_entries[entry] = surf_pbx_entries
        return pbx_entries

    # 2D surface Pourbaix diagram case
    def _construct_entry_hyperplanes(self, domain: PourbaixEntry) -> np.ndarray:
        """Construct the entry hyperplanes for each stable domain.

        Args:
            domain: Stable domain for which to construct the hyperplanes.

        Returns:
            np.ndarray of surface Pourbaix entry-based hyperplanes for the domain.
        """
        pourbaix_entries = self.ind_surface_pbx_entries.get(domain)
        hyperplanes = np.array(
            [
                np.array([-PREFAC * entry.npH, -entry.nPhi, 0, -entry.energy])
                * entry.normalization_factor
                for entry in pourbaix_entries
            ]
        )
        hyperplanes[:, 2] = 1
        return hyperplanes

    # 3D surface Pourbaix diagram case
    def _construct_3D_entry_hyperplanes(
        self,
        domain: PourbaixEntry,
        at_equilibrium: bool = False,
        vertices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct the entry hyperplanes for each stable domain, with additional log(conc) term.

        Args:
            domain: Stable domain for which to construct the hyperplanes.
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium
                conditions or not. Default is False.
            vertices: Boundaries in the pH-V-log(conc) space of the stable domain. Default is None.

        Returns:
            np.ndarray of surface Pourbaix entry-based hyperplanes for the domain.
        """
        pourbaix_entries = self.ind_3D_surface_pbx_entries.get(domain)
        hyperplanes = np.array(
            [
                np.array(
                    [
                        -PREFAC * entry.npH,
                        -entry.nPhi,
                        -PREFAC * entry.n_conc,
                        0,
                        -entry.energy_without_conc_term,
                    ]
                )
                * entry.normalization_factor
                for entry in pourbaix_entries
            ]
        )
        hyperplanes[:, -2] = 1

        #     # TODO: Write test cases here
        return hyperplanes

    # 2D surface Pourbaix diagram case
    def _construct_border_hyperplanes_and_interior_point(
        self, vertices: np.ndarray, entry_hyperplanes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # TODO: add test cases
        """Construct the border hyperplanes and calculate an interior point for each stable domain.

        Args:
            vertices: Boundaries in the pH-V space of the stable domain.
            entry_hyperplanes: Surface Pourbaix entry-based hyperplanes calculated for the domain.

        Returns:
            tuple of border hyperplanes and interior point for the domain.
        """
        convex_hull = ConvexHull(vertices)
        border_ineqs = np.insert(convex_hull.equations, 2, [0] * len(convex_hull.equations), axis=1)
        limits = np.vstack([np.min(vertices, axis=0), np.max(vertices, axis=0)]).T
        g_max = PourbaixDiagram.get_min_energy(limits, entry_hyperplanes)
        lower_bound_ineq = [0, 0, -1, 2 * g_max]
        border_hyperplanes = np.vstack([border_ineqs, [lower_bound_ineq]])
        interior_point = np.array([*np.mean(vertices, axis=0).tolist(), g_max])
        return border_hyperplanes, interior_point

    # 3D surface Pourbaix diagram case
    def _construct_3D_border_hyperplanes_and_interior_point(
        self, vertices: np.ndarray, entry_hyperplanes: np.ndarray, at_equilibrium: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the border hyperplanes with an additional log(conc) term and calculate an interior point for each
        stable domain.

        Args:
            vertices: Boundaries in the pH-V-log(conc) space of the stable domain.
            entry_hyperplanes: Surface Pourbaix entry-based hyperplanes calculated for the domain.
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium

        Returns:
            tuple of border hyperplanes and interior point for the domain.
        """
        # if at_equilibrium:
        convex_hull = ConvexHull(vertices, qhull_options="QJ")
        # else:
        #     convex_hull = ConvexHull(vertices)
        border_ineqs = np.insert(
            convex_hull.equations, -1, [0] * len(convex_hull.equations), axis=1
        )
        limits = np.vstack([np.min(vertices, axis=0), np.max(vertices, axis=0)]).T
        g_max = PourbaixDiagram.get_min_energy(limits, entry_hyperplanes)
        lower_bound_ineq = [0, 0, 0, -1, 2 * g_max]
        border_hyperplanes = np.vstack([border_ineqs, [lower_bound_ineq]])
        interior_point = np.array([*np.mean(vertices, axis=0).tolist(), g_max])
        # NOTE: not needed anymore, but keeping the code here for reference
        # # For the at equilibrium case, we can project the interior_point onto the equilibrium plane
        # # To do this, we need to find the normal vector of the plane and the point on the plane
        # if at_equilibrium:
        #     # Find equation of the plane in 3D space to express lg(conc) as a function of pH and V
        #     _points, _v, w = simple_pca(vertices, k=3)
        #     normal_vec = w[:, 2]  # get the normal vector of the projected (2D) plane
        #     point_on_plane = np.mean(vertices, axis=0)

        #     dx = interior_point[:2] - point_on_plane[:2]

        #     # Project the interior point onto the plane
        #     interior_point[:2] = point_on_plane[:2] + np.dot(dx, normal_vec[:2]) * normal_vec[:2]
        return border_hyperplanes, interior_point

    # Both 2D and 3D surface Pourbaix diagram case
    def construct_hyperplanes(
        self,
        process_3D: bool = False,
        at_equilibrium: bool = False,
    ) -> dict[PourbaixEntry, dict[str, np.ndarray]]:
        """Construct the hyperplanes and obtain an interior point for each stable domain.

        Each hyperplane represents an inequality of the form a*pH + b*V + c*z - g(0, 0) <= 0 in the pH-V-z space,
        where z represents an additional dimension that allows the energy to vary with pH and V. The hyperplanes
        enclose the stable domain. If process_3D is True, the hyperplanes will also include an additional log(conc) term.

        The interior point is a point inside the stable domain.

        Args:
            process_3D (bool): whether we are processing a 3D Pourbaix diagram. Default is False.
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium
                conditions. Default is False.

        Returns:
            dictionary of hyperplanes and interior points for each stable domain.
        """
        hyperplanes = {}
        stable_vertices = (
            self.ref_pbx.stable_3D_vertices if process_3D else self.ref_pbx.stable_vertices
        )
        for entry, vertices in stable_vertices.items():
            if process_3D:
                entry_hyperplanes = self._construct_3D_entry_hyperplanes(
                    entry, at_equilibrium=at_equilibrium, vertices=vertices
                )
                border_hyperplanes, interior_point = (
                    self._construct_3D_border_hyperplanes_and_interior_point(
                        vertices, entry_hyperplanes, at_equilibrium=at_equilibrium
                    )
                )

            else:
                entry_hyperplanes = self._construct_entry_hyperplanes(entry)
                border_hyperplanes, interior_point = (
                    self._construct_border_hyperplanes_and_interior_point(
                        vertices, entry_hyperplanes
                    )
                )
            hyperplanes[entry] = {
                "hyperplanes": np.vstack([entry_hyperplanes, border_hyperplanes]),
                "interior_point": interior_point,
            }
            logger.debug("Entry: %s, Hyperplanes: %s", entry, hyperplanes[entry]["hyperplanes"])
            logger.debug("vertices: %s", vertices)
        return hyperplanes

    # TODO: move to PourbaixDiagram
    @classmethod
    def _sort_pourbaix_domain_vertices(cls, points: Iterable[float]) -> np.ndarray:
        """Sort the Pourbaix domain vertices in the pH and V (phi) axes.

        Args:
            points: Iterable of points to sort.

        Returns:
            np.ndarray of sorted points.
        """
        points = np.array(points)
        # Initial sort to ensure consistency
        points = points[np.lexsort(np.transpose(points))]
        center = np.mean(points, axis=0)
        points_centered = points - center

        # Sort points in a clockwise manner
        angles = np.arctan2(points_centered[:, 1], points_centered[:, 0])
        points_centered = points_centered[np.argsort(angles)]
        return points_centered + center

    # Both 2D and 3D surface Pourbaix diagram case
    def _get_pourbaix_domains_ind(
        self,
        domain: PourbaixEntry,
        halfspace_int: HalfspaceIntersection,
        process_3D: bool = False,
        at_equilibrium: bool = False,
    ) -> tuple[dict, dict]:
        """
        Construct and sort the surface Pourbaix domains for an original stable domain.

        Args:
            domain: Stable domain for which to obtain surface Pourbaix entries.
            halfspace_int: HalfspaceIntersection object for the domain.
            process_3D: whether we are processing a 3D Pourbaix diagram
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium

        Returns:
            dictionary of Pourbaix domains of Simplex objects and dictionary of sorted vertices for
            specified original domain.
        """
        if process_3D:
            pourbaix_entries = self.ind_3D_surface_pbx_entries.get(domain)
        else:
            pourbaix_entries = self.ind_surface_pbx_entries.get(domain)
        pourbaix_domain_vertices = defaultdict(list)
        # Attribute intersection points to surface Pourbaix entries, which define the Pourbaix domain for each entry
        for intersection, facet in zip(halfspace_int.intersections, halfspace_int.dual_facets):
            for v in facet:
                if v < len(pourbaix_entries):  # remove the lower bound
                    this_entry = pourbaix_entries[v]
                    pourbaix_domain_vertices[this_entry].append(intersection)

        pourbaix_domains = {}
        pourbaix_domain_sorted_vertices = {}
        for entry, points in pourbaix_domain_vertices.items():
            points = np.array(points)[:, :3] if process_3D else np.array(points)[:, :2]
            # Sorting might not be important for the 3D case
            sorted_points = self._sort_pourbaix_domain_vertices(points)

            if process_3D and at_equilibrium:
                points_2d, _v, w = simple_pca(sorted_points, k=2)
                domain = ConvexHull(points_2d)
                centroid_2d = get_centroid_2d(points_2d[domain.vertices])
                ann_loc = centroid_2d @ w.T + np.mean(sorted_points.T, axis=1)
                simplices = [Simplex(sorted_points[indices]) for indices in domain.simplices]
                vertices = sorted_points[domain.vertices]  # ordered in 2D space
                pourbaix_domains[entry] = simplices
                pourbaix_domain_sorted_vertices[entry] = vertices
            else:
                pourbaix_domains[entry] = [
                    Simplex(sorted_points[indices])
                    for indices in ConvexHull(
                        sorted_points
                    ).simplices  # convex hull only to get simplices
                ]
                pourbaix_domain_sorted_vertices[entry] = sorted_points
        return pourbaix_domains, pourbaix_domain_sorted_vertices

    # Both 2D and 3D surface Pourbaix diagram case
    def construct_pourbaix_domains(
        self, process_3D: bool = False, at_equilibrium: bool = False
    ) -> tuple[dict, dict]:
        """Obtain the surface Pourbaix domains for each original stable domain.

        Args:
            process_3D: whether we are processing a 3D Pourbaix diagram
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium

        Returns:
            dictionary of Pourbaix domains of Simplex objects and dictionary of sorted vertices for each domain.
        """
        stable_domains = {}
        stable_domain_vertices = {}
        ind_hyperplanes = self.ind_3D_hyperplanes if process_3D else self.ind_hyperplanes
        for entry, hyperplane_info in ind_hyperplanes.items():
            logger.debug("Entry: %s, Hyperplanes: %s", entry, hyperplane_info["hyperplanes"])
            hyperplanes = hyperplane_info["hyperplanes"]
            interior_point = hyperplane_info["interior_point"]
            hs_int = HalfspaceIntersection(hyperplanes, interior_point)
            pourbaix_domains, pourbaix_domain_vertices = self._get_pourbaix_domains_ind(
                entry, hs_int, process_3D=process_3D, at_equilibrium=at_equilibrium
            )
            stable_domains[entry] = pourbaix_domains
            stable_domain_vertices[entry] = pourbaix_domain_vertices
        return stable_domains, stable_domain_vertices

    # Both 2D and 3D surface Pourbaix diagram case
    def merge_pourbaix_domains(
        self, process_3D: bool = False, at_equilibrium: bool = False
    ) -> tuple[dict, dict]:
        """Construct the overall Pourbaix domains across all original stable domains.

        Args:
            process_3D: whether we are processing a 3D Pourbaix diagram
            at_equilibrium (bool): whether to calculate the Pourbaix diagram at equilibrium

        Returns:
            dictionary of Pourbaix domains of Simplex objects and dictionary of sorted vertices.
        """
        ind_stable_domains = self.ind_3D_stable_domains if process_3D else self.ind_stable_domains
        ind_stable_domain_vertices = (
            self.ind_3D_stable_domain_vertices if process_3D else self.ind_stable_domain_vertices
        )

        # Merge the stable domain vertices
        merged_stable_domains = defaultdict(list)
        merged_stable_domain_vertices = defaultdict(list)
        for bulk_pbx_entry, domain_vertices in ind_stable_domain_vertices.items():
            for surf_pbx_entry, vertices in domain_vertices.items():
                merged_stable_domains[surf_pbx_entry.entry] += ind_stable_domains[bulk_pbx_entry][
                    surf_pbx_entry
                ]
                merged_stable_domain_vertices[surf_pbx_entry.entry] += vertices.tolist()

        # TODO make it contain more information about the surface for disambiguation
        merged_pbx_stable_domains = {}
        merged_pbx_stable_domain_vertices = {}
        for k, v_simplex in merged_stable_domains.items():
            pbx_entry = PourbaixEntry(k)
            # merged_stable_domains[k] = list(set(v_simplex))
            merged_pbx_stable_domains[pbx_entry] = list(set(v_simplex))
            v_vertices = merged_stable_domain_vertices[k]
            merged_pbx_stable_domain_vertices[pbx_entry] = np.unique(
                np.array(v_vertices).round(decimals=3),
                axis=0,  # round to avoid numerical errors
            )

        # Create fake Pourbaix entries for each region to be compatible with PourbaixDiagram
        # merged_stable_domains = {PourbaixEntry(k): v for k, v in merged_stable_domains.items()}
        # merged_stable_domain_vertices = {PourbaixEntry(k): v for k, v in merged_stable_domain_vertices.items()}

        merged_pbx_stable_sorted_vertices = {}

        # General idea: for overlapping points between different entries, keep them
        # For remaining points, do convex hull to remove interior points, and then add back the overlapping points

        # Might not be useful for the 3D case
        all_points = np.concatenate(list(merged_pbx_stable_domain_vertices.values())).round(
            decimals=3
        )
        all_points = np.array(all_points)[:, :3] if process_3D else np.array(all_points)[:, :2]
        all_points = self._sort_pourbaix_domain_vertices(all_points)
        unique_points, counts = np.unique(all_points, return_counts=True, axis=0)
        overlapping_points = unique_points[counts > 1]

        for entry, points in merged_pbx_stable_domain_vertices.items():
            points = np.array(points)[:, :3] if process_3D else np.array(points)[:, :2]
            sorted_points = np.unique(
                self._sort_pourbaix_domain_vertices(points).round(decimals=3), axis=0
            )

            # Check if sorted points contain overlapping points
            overlapping_indices = np.where(
                np.isclose(sorted_points[:, None, :], overlapping_points[None, :, :], atol=1e-3)
                .all(-1)
                .any(-1)
            )[0]
            if process_3D and len(sorted_points) < 4:
                # Can't do convex hull with less than 4 points
                hull_idx = np.arange(len(sorted_points)).tolist()
                print("Less than 4 points, using all points, indices: ", hull_idx)
            else:
                hull = ConvexHull(sorted_points)  # convex hull to remove interior points
                hull_idx = hull.vertices.tolist()

            # Combine the convex hull vertices with the overlapping points
            combined_idx = np.unique(sorted(overlapping_indices.tolist() + hull_idx))
            # Rearrange the indices again for plotting
            merged_pbx_stable_sorted_vertices[entry] = self._sort_pourbaix_domain_vertices(
                sorted_points[combined_idx]
            )

        return merged_pbx_stable_domains, merged_pbx_stable_sorted_vertices

    def get_all_entries_at_conditions(
        self,
        pH: float,
        V: float,
        reference_entry_id: str = None,
        subset_entry_ids: list[str] = None,
    ) -> tuple[PourbaixEntry, dict]:
        """Get all SurfacePourbaixEntries at a given pH and V condition.

        Args:
            pH: pH at which to find the entries.
            V: V at which to find the entries.
            reference_entry: reference entry id to normalize the energies.
            subset_entry_ids: list of entry ids to plot.

        Returns:
            tuple of stable entry and dictionary of all energies at the given conditions
        """
        # if reference_entry:
        #     reference_energy = reference_entry.normalized_energy_at_conditions(pH, V)
        # else:
        #     reference_energy = 0
        stable_entry = self.ref_pbx.get_stable_entry(
            pH, V
        )  # regular PourbaixEntry at the given pH and V
        surface_pbx_entries = self.ind_surface_pbx_entries.get(
            stable_entry
        )  # SurfacePourbaixEntries associated with the regular PourbaixEntry
        all_energies = {}
        if reference_entry_id:
            reference_energy = next(
                entry.normalized_energy_at_conditions(pH, V)
                for entry in surface_pbx_entries
                if entry.entry_id == reference_entry_id
            )
        else:
            reference_energy = 0
        if subset_entry_ids is not None:
            surface_pbx_entries = [
                entry for entry in surface_pbx_entries if entry.entry_id in subset_entry_ids
            ]
        for surface_pbx_entry in surface_pbx_entries:
            all_energies[surface_pbx_entry.surface_entry] = (
                surface_pbx_entry.normalized_energy_at_conditions(pH, V)
            )
            all_energies[surface_pbx_entry.surface_entry] -= reference_energy
        return stable_entry, all_energies

    def as_dict(self):
        """Get MSONable dict."""
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "surface_entries": [entry.as_dict() for entry in self.surface_entries],
            "reference_pourbaix_diagram": self.ref_pbx.as_dict(),
            "reference_elements": self.ref_elems,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            SurfacePourbaixDiagram
        """
        return cls(
            MontyDecoder().process_decoded(dct["surface_entries"]),
            MontyDecoder().process_decoded(dct["reference_pourbaix_diagram"]),
            reference_elements=dct["reference_elements"],
        )


class PourbaixPlotter:
    """A plotter class for phase diagrams."""

    def __init__(self, pourbaix_diagram):
        """
        Args:
            pourbaix_diagram (PourbaixDiagram): A PourbaixDiagram object.
        """
        self._pbx = pourbaix_diagram

    def show(self, *args, **kwargs):
        """Show the Pourbaix plot.

        Args:
            *args: args to get_pourbaix_plot
            **kwargs: kwargs to get_pourbaix_plot
        """
        plt = self.get_pourbaix_plot(*args, **kwargs)
        plt.show()

    def get_3D_pourbaix_plot(self):
        pass

    @no_type_check
    def get_pourbaix_plot(
        self,
        limits: tuple[float, float] | None = None,
        title: str = "",
        label_domains: bool = True,
        label_fontsize: int = 20,
        show_water_lines: bool = True,
        show_neutral_axes: bool = True,
        cmap: str | Colormap = "RdYlBu_r",
        cmap_values: dict[str, float] | None = None,
        ax: plt.Axes = None,
        lw: int = 2,
        full_formula: bool = False,
    ) -> plt.Axes:
        """
        Plot Pourbaix diagram.

        Args:
            limits: 2D list containing limits of the Pourbaix diagram
                of the form [[xlo, xhi], [ylo, yhi]]
            title (str): Title to display on plot
            label_domains (bool): whether to label Pourbaix domains
            label_fontsize: font size for domain labels
            show_water_lines: whether to show dashed lines indicating the region
                of water stability.
            show_neutral_axes; whether to show dashed horizontal and vertical lines
                at 0 V and pH 7, respectively.
            cmap (str or Colormap): colormap for Pourbaix diagram
            cmap_values (dict): dictionary of values to use for colormap
            ax (Axes): Matplotlib Axes instance for plotting
            lw (int): Line width for each Pourbaix domain
            full_formula (bool): Whether to display full formula for each entry

        Returns:
            Axes: matplotlib Axes object with Pourbaix diagram
        """
        if limits is None:
            limits = [[-2, 16], [-3, 3]]

        ax = ax or pretty_plot(16)

        xlim, ylim = limits

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        if show_water_lines:
            h_line = np.transpose([[xlim[0], -xlim[0] * PREFAC], [xlim[1], -xlim[1] * PREFAC]])
            o_line = np.transpose(
                [
                    [xlim[0], -xlim[0] * PREFAC + 1.23],
                    [xlim[1], -xlim[1] * PREFAC + 1.23],
                ]
            )
            ax.plot(h_line[0], h_line[1], "r--", linewidth=lw)
            ax.plot(o_line[0], o_line[1], "r--", linewidth=lw)

        if show_neutral_axes:
            neutral_line = np.transpose([[7, ylim[0]], [7, ylim[1]]])
            V0_line = np.transpose([[xlim[0], 0], [xlim[1], 0]])
            ax.plot(neutral_line[0], neutral_line[1], "k-.", linewidth=lw)
            ax.plot(V0_line[0], V0_line[1], "k-.", linewidth=lw)

        # Sort according to formula string
        # self._pbx.stable_entries.sort(lambda x: x.entry_id)

        for entry in self._pbx.stable_entries:
            vertices = self._pbx.stable_vertices[entry]
            # for entry, vertices in self._pbx._stable_domain_vertices.items():
            center = np.mean(vertices, axis=0)
            x, y = np.transpose(np.vstack([vertices, vertices[0]]))
            ax.plot(x, y, "k-", linewidth=lw)

            if cmap_values is not None:
                cmap_value = cmap_values[entry.entry_id]
            else:
                # Use reduced formula to index a number for the color
                cmap_value = (
                    np.sum([ord(c) for c in entry.composition.reduced_formula]) % 128
                ) / 128

            # normalized_dist_origin = np.linalg.norm(center) / np.linalg.norm([xlim[1], ylim[1]])
            # normalized_dist_center = np.linalg.norm(center - np.mean(limits)) / np.linalg.norm(limits)
            ax.fill(x, y, alpha=0.75, color=cmap(cmap_value))

            if label_domains:
                ax.annotate(
                    generate_entry_label(entry, full_formula=full_formula),
                    center,
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color="k",
                ).draggable()

        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set(xlabel="pH", ylabel=r"$U_{SHE}$ (V)", xlim=xlim, ylim=ylim)
        return ax

    @no_type_check
    def plot_entry_stability(
        self,
        entry: Any,
        pH_range: tuple[float, float] = (-2, 16),
        pH_resolution: int = 100,
        V_range: tuple[float, float] = (-3, 3),
        V_resolution: int = 100,
        e_hull_max: float = 1,
        cmap: str = "RdYlBu_r",
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """
        Plots the stability of an entry in the Pourbaix diagram.

        Args:
            entry (Any): The entry to plot stability for.
            pH_range (tuple[float, float], optional): pH range for the plot. Defaults to (-2, 16).
            pH_resolution (int, optional): pH resolution. Defaults to 100.
            V_range (tuple[float, float], optional): Voltage range for the plot. Defaults to (-3, 3).
            V_resolution (int, optional): Voltage resolution. Defaults to 100.
            e_hull_max (float, optional): Maximum energy above the hull. Defaults to 1.
            cmap (str, optional): Colormap for the plot. Defaults to "RdYlBu_r".
            ax (Axes, optional): Existing matplotlib Axes object for plotting. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to `get_pourbaix_plot`.

        Returns:
            plt.Axes: Matplotlib Axes object with the plotted stability.
        """
        # Plot the Pourbaix diagram
        ax = self.get_pourbaix_plot(ax=ax, **kwargs)
        pH, V = np.mgrid[
            pH_range[0] : pH_range[1] : pH_resolution * 1j,
            V_range[0] : V_range[1] : V_resolution * 1j,
        ]

        stability = self._pbx.get_decomposition_energy(entry, pH, V)

        # Plot stability map
        cax = ax.pcolor(pH, V, stability, cmap=cmap, vmin=0, vmax=e_hull_max)
        cbar = ax.figure.colorbar(cax)
        full_formula = kwargs.get("full_formula", False)
        cbar.set_label(
            f"Stability of {generate_entry_label(entry, full_formula=full_formula)} (eV/atom)"
        )

        # Set ticklabels
        # ticklabels = [t.get_text() for t in cbar.ax.get_yticklabels()]
        # ticklabels[-1] = f">={ticklabels[-1]}"
        # cbar.ax.set_yticklabels(ticklabels)

        return ax

    def get_energy_vs_potential_plot(
        self,
        pH,
        energy_range: Tuple[int] = None,
        reference_entry_id: SurfacePourbaixEntry = None,
        subset_entry_ids: list[str] = None,
        show_non_subset_entries: bool = False,
        highlight_subset_entries: bool = False,
        V_range: tuple[float, float] = (-1, 2),
        V_resolution: int = 100,
        cmap: str = "RdYlBu_r",
        cmap_values: dict[str, float] = None,
        ax=None,
        lw=2,
        full_formula=False,
        label_fontsize: int = 20,
        label_domains: bool = True,
        label_domain_center: float = 0.5,
        label_domain_positions: dict[str, float] = None,
        label_stable_bulk: bool = True,
        color_stable_bulk: bool = True,
        bulk_cmap_values: dict[str, float] = None,
    ) -> plt.Axes:
        """Get the energy of an entry at a given pH as a function of potential.

        Args:
            pH: pH at which to get energy
            energy_range (tuple[int], optional): energy limits for the plot. Defaults to None.
            reference_entry_id: reference entry id to offset the energies
            subset_entry_ids: list of entry ids to plot
            show_non_subset_entries: whether to show non-subset entries
            highlight_subset_entries: whether to highlight the subset entries
            V_range (tuple[float, float], optional): Voltage range for the plot. Defaults to (-3, 3).
            V_resolution (int, optional): Voltage resolution. Defaults to 100.
            cmap (str, optional): Colormap for the plot. Defaults to "RdYlBu_r".
            cmap_values (dict[str, float], optional): Dictionary of values to use for colormap. Defaults to None.
            ax (Axes, optional): Existing matplotlib Axes object for plotting. Defaults to None.
            lw (int, optional): Line width for the plot. Defaults to 2.
            full_formula (bool, optional): Whether to use full formula for the entry. Defaults to False.
            label_fontsize (int, optional): Font size for the labels. Defaults to 20.
            label_domains (bool, optional): Whether to label the domains. Defaults to True.
            label_domain_center (float, optional): Center for the domain labels. Defaults to 0.5.
            label_domain_positions (dict[str, float], optional): Positions for the domain labels. Defaults to None.
            label_stable_bulk (bool, optional): Whether to label the stable bulk. Defaults to True
            color_stable_bulk (bool, optional): Whether to color the stable bulk. Defaults to True
            bulk_cmap_values (dict[str, float], optional): Dictionary of values to use for colormap for the stable bulk.

        Returns:
            plt.Axes: Matplotlib Axes object with the energy plot
        """
        ax = ax or pretty_plot(16)

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        all_Vs = np.linspace(V_range[0], V_range[1], V_resolution)
        all_energies = defaultdict(list)
        all_stable_bulks_for_range = []

        if subset_entry_ids is not None and not show_non_subset_entries:
            included_subset_entry_ids = subset_entry_ids
        else:
            included_subset_entry_ids = None

        # Obtain all energies for each entry at the given pH and V
        for V in all_Vs:
            stable_entry, curr_all_energies = self._pbx.get_all_entries_at_conditions(
                pH,
                V,
                reference_entry_id=reference_entry_id,
                subset_entry_ids=included_subset_entry_ids,
            )
            all_stable_bulks_for_range.append(stable_entry)
            for entry, energy in curr_all_energies.items():
                all_energies[entry].append(energy)

        # Plot all entry with energies
        # Sort the entries by label
        for i, (entry, energies) in enumerate(all_energies.items()):
            if not isinstance(entry, PourbaixEntry):
                entry = PourbaixEntry(entry)
            if highlight_subset_entries and entry.entry_id not in subset_entry_ids:
                ax.plot(all_Vs, energies, color="gray", linewidth=lw / 2, alpha=0.2)
            else:
                if cmap_values is not None:
                    cmap_value = cmap_values[entry.entry_id]
                else:
                    cmap_value = (
                        np.sum([ord(c) for c in entry.composition.reduced_formula]) % 128
                    ) / 128
                ax.plot(
                    all_Vs,
                    energies,
                    label=generate_entry_label(entry, full_formula=full_formula),
                    linewidth=lw,
                    alpha=0.75,
                    zorder=5,
                    color=cmap(cmap_value),
                )
                if (
                    label_domain_positions is not None
                    and label_domain_positions.get(entry.entry_id) is not None
                ):
                    center = label_domain_positions[entry.entry_id]
                else:
                    center = (
                        all_Vs[int(label_domain_center * V_resolution)],
                        energies[int(label_domain_center * V_resolution)],
                    )
                if label_domains:
                    ax.annotate(
                        generate_entry_label(entry, full_formula=full_formula),
                        center,
                        ha="center",
                        va="center",
                        fontsize=label_fontsize,
                        color="k",
                        zorder=11,
                    ).draggable()

        # Find the region for each stable bulk
        if color_stable_bulk:
            curr_stable_bulk = all_stable_bulks_for_range[0]
            prev_index = 0
            for i, stable_bulk in enumerate(all_stable_bulks_for_range):
                if bulk_cmap_values is not None:
                    bulk_cmap_value = bulk_cmap_values[curr_stable_bulk.entry_id]
                else:
                    bulk_cmap_value = (
                        np.sum([ord(c) for c in curr_stable_bulk.composition.reduced_formula]) % 128
                    ) / 128
                if stable_bulk != curr_stable_bulk:
                    ax.fill_between(
                        all_Vs[prev_index : i + 1],
                        energy_range[0] if energy_range else min(energies),
                        energy_range[1] if energy_range else max(energies),
                        alpha=0.1,
                        color=cmap(bulk_cmap_value),
                    )
                    # Annotate the stable bulk flipped 90 degrees counter-clockwise to the plot
                    center = (
                        all_Vs[(prev_index + i) // 2],
                        (energy_range[0] if energy_range else min(energies)) + 0.25,
                    )
                    if label_stable_bulk:
                        ax.annotate(
                            generate_entry_label(curr_stable_bulk, full_formula=False),
                            center,
                            ha="center",
                            va="bottom",
                            fontsize=0.75 * label_fontsize,
                            color="k",
                            alpha=0.5,
                            rotation=90,
                        ).draggable()
                    # Update the current stable bulk and the previous index
                    curr_stable_bulk = stable_bulk
                    prev_index = i
            if bulk_cmap_values is not None:
                bulk_cmap_value = bulk_cmap_values[curr_stable_bulk.entry_id]
            else:
                bulk_cmap_value = (
                    np.sum([ord(c) for c in curr_stable_bulk.composition.reduced_formula]) % 128
                ) / 128
            ax.fill_between(
                all_Vs[prev_index:],
                energy_range[0] if energy_range else min(energies),
                energy_range[1] if energy_range else max(energies),
                alpha=0.1,
                color=cmap(bulk_cmap_value),
            )
            center = (
                all_Vs[(prev_index + len(all_stable_bulks_for_range)) // 2],
                (energy_range[0] if energy_range else min(energies)) + 0.25,
            )
            if label_stable_bulk:
                ax.annotate(
                    generate_entry_label(curr_stable_bulk, full_formula=False),
                    center,
                    ha="right",
                    va="bottom",
                    fontsize=0.75 * label_fontsize,
                    color="k",
                    alpha=0.5,
                    rotation=90,
                ).draggable()

        # ax.legend()
        ax.set_xlim(V_range)
        if energy_range:
            ax.set_ylim(energy_range)
        # ax.set_title(r"$\Delta$ Energy vs Potential at pH " + f"{pH}", fontsize=20, fontweight="bold")
        ax.set(xlabel=r"$U_{SHE}$ (V)", ylabel=r"$\Delta\Omega$ (eV/unit surface)")
        return ax

    def domain_vertices(self, entry):
        """Get the vertices of the Pourbaix domain.

        Args:
            entry: Entry for which domain vertices are desired

        Returns:
            list of vertices
        """
        return self._pbx._stable_domain_vertices[entry]


def generate_entry_label(
    entry, full_formula: bool = False, style: Literal["latex", "html"] = "latex"
) -> str:
    """
    Generates a label for the Pourbaix plotter.

    Args:
        entry (PourbaixEntry or MultiEntry): entry to get a label for
        full_formula (bool): whether to use full formula
        style (str): style of the label

    Returns:
        str: label for the entry
    """
    if isinstance(entry, MultiEntry):
        sorted_entry_list = sorted(entry.entry_list, key=lambda x: x.name)
        return " + ".join(
            generate_entry_label(sub_entry, full_formula=full_formula, style=style)
            for sub_entry in sorted_entry_list
        )

    # TODO - a more elegant solution could be added later to Stringify
    # for example, the pattern re.sub(r"([-+][\d\.]*)", r"$^{\1}$", )
    # will convert B(OH)4- to B(OH)$_4^-$.
    # for this to work, the ion's charge always must be written AFTER
    # the sign (e.g., Fe+2 not Fe2+)
    if style == "html":
        string = entry.to_html_string(full_formula=full_formula)
        return re.sub(
            r"()\[([^)]*)\]", r"\1<sup>\2</sup>", string
        )  # replace [ with <sup> for superscript
    string = entry.to_latex_string(full_formula=full_formula)
    return re.sub(r"()\[([^)]*)\]", r"\1$^{\2}$", string)  # replace [ with ^{ for superscript
