"""
Module for reading Lobster input files. For more information
on LOBSTER see www.cohp.de.
If you use this module, please cite:
J. George, G. Petretto, A. Naik, M. Esters, A. J. Jackson, R. Nelson, R. Dronskowski, G.-M. Rignanese, G. Hautier,
"Automated Bonding Analysis with Crystal Orbital Hamilton Populations",
ChemPlusChem 2022, e202200123,
DOI: 10.1002/cplu.202200123.
"""

from __future__ import annotations

import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING

import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self

    from pymatgen.core.composition import Composition
    from pymatgen.util.typing import Tuple3Ints


__author__ = "Janine George, Marco Esters"
__copyright__ = "Copyright 2017, The Materials Project"
__version__ = "0.2"
__maintainer__ = "Janine George"
__email__ = "janinegeorge.ulfen@gmail.com"
__date__ = "Dec 13, 2017"

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


due.cite(
    Doi("10.1002/cplu.202200123"),
    description="Automated Bonding Analysis with Crystal Orbital Hamilton Populations",
)


class Lobsterin(UserDict, MSONable):
    """
    This class can handle and generate lobsterin files
    Furthermore, it can also modify INCAR files for lobster, generate KPOINT files for fatband calculations in Lobster,
    and generate the standard primitive cells in a POSCAR file that are needed for the fatband calculations.
    There are also several standard lobsterin files that can be easily generated.
    """

    # reminder: lobster is not case sensitive

    # keyword + one float can be used in file
    FLOAT_KEYWORDS = (
        "COHPstartEnergy",
        "COHPendEnergy",
        "gaussianSmearingWidth",
        "useDecimalPlaces",
        "COHPSteps",
    )
    # one of these keywords +endstring can be used in file
    STRING_KEYWORDS = (
        "basisSet",
        "cohpGenerator",
        "realspaceHamiltonian",
        "realspaceOverlap",
        "printPAWRealSpaceWavefunction",
        "printLCAORealSpaceWavefunction",
        "kSpaceCOHP",
        "EwaldSum",
    )
    # the keyword alone will turn on or off a function
    BOOLEAN_KEYWORDS = (
        "saveProjectionToFile",
        "skipdos",
        "skipcohp",
        "skipcoop",
        "skipcobi",
        "skipMadelungEnergy",
        "loadProjectionFromFile",
        "forceEnergyRange",
        "DensityOfEnergy",
        "BWDF",
        "BWDFCOHP",
        "skipPopulationAnalysis",
        "skipGrossPopulation",
        "userecommendedbasisfunctions",
        "skipProjection",
        "writeBasisFunctions",
        "writeMatricesToFile",
        "noFFTforVisualization",
        "RMSp",
        "onlyReadVasprun.xml",
        "noMemoryMappedFiles",
        "skipPAWOrthonormalityTest",
        "doNotIgnoreExcessiveBands",
        "doNotUseAbsoluteSpilling",
        "skipReOrthonormalization",
        "forceV1HMatrix",
        "useOriginalTetrahedronMethod",
        "forceEnergyRange",
        "bandwiseSpilling",
        "kpointwiseSpilling",
        "LSODOS",
    )
    # several of these keywords + ending can be used in a lobsterin file:
    LISTKEYWORDS = ("basisfunctions", "cohpbetween", "createFatband")

    # all keywords known to this class so far
    AVAILABLE_KEYWORDS = FLOAT_KEYWORDS + STRING_KEYWORDS + BOOLEAN_KEYWORDS + LISTKEYWORDS

    def __init__(self, settingsdict: dict):
        """
        Args:
            settingsdict: dict to initialize Lobsterin.
        """
        super().__init__()
        # check for duplicates
        keys = [key.lower() for key in settingsdict]
        if len(keys) != len(set(keys)):
            raise KeyError("There are duplicates for the keywords!")
        self.update(settingsdict)

    def __setitem__(self, key, val) -> None:
        """
        Add parameter-val pair to Lobsterin. Warns if parameter is not in list of
        valid lobsterin tags. Also cleans the parameter and val by stripping
        leading and trailing white spaces. Similar to INCAR class.
        """
        # due to the missing case sensitivity of lobster, the following code is necessary
        new_key = next((key_here for key_here in self if key.strip().lower() == key_here.lower()), key)

        if new_key.lower() not in [element.lower() for element in Lobsterin.AVAILABLE_KEYWORDS]:
            raise KeyError("Key is currently not available")

        super().__setitem__(new_key, val.strip() if isinstance(val, str) else val)

    def __getitem__(self, key) -> Any:
        """Implements getitem from dict to avoid problems with cases."""
        normalized_key = next((k for k in self if key.strip().lower() == k.lower()), key)

        key_is_unknown = normalized_key.lower() not in map(str.lower, Lobsterin.AVAILABLE_KEYWORDS)
        if key_is_unknown or normalized_key not in self.data:
            raise KeyError(f"{key=} is not available")

        return self.data[normalized_key]

    def __contains__(self, key) -> bool:
        """Implements getitem from dict to avoid problems with different key casing."""
        normalized_key = next((k for k in self if key.strip().lower() == k.lower()), key)

        key_is_unknown = normalized_key.lower() not in map(str.lower, Lobsterin.AVAILABLE_KEYWORDS)
        return not key_is_unknown and normalized_key in self.data

    def __delitem__(self, key):
        new_key = next((key_here for key_here in self if key.strip().lower() == key_here.lower()), key)

        del self.data[new_key]

    def diff(self, other):
        """
        Diff function for lobsterin. Compares two lobsterin and indicates which parameters are the same.
        Similar to the diff in INCAR.

        Args:
            other (Lobsterin): Lobsterin object to compare to

        Returns:
            dict with differences and similarities
        """
        similar_param = {}
        different_param = {}
        key_list_others = [element.lower() for element in other]

        for k1, v1 in self.items():
            k1_lower = k1.lower()
            k1_in_other = next((key_here for key_here in other if key_here.lower() == k1_lower), k1_lower)
            if k1_lower not in key_list_others:
                different_param[k1.lower()] = {"lobsterin1": v1, "lobsterin2": None}
            elif isinstance(v1, str):
                if v1.strip().lower() != other[k1_lower].strip().lower():
                    different_param[k1.lower()] = {
                        "lobsterin1": v1,
                        "lobsterin2": other[k1_in_other],
                    }
                else:
                    similar_param[k1.lower()] = v1
            elif isinstance(v1, list):
                new_set1 = {element.strip().lower() for element in v1}
                new_set2 = {element.strip().lower() for element in other[k1_in_other]}
                if new_set1 != new_set2:
                    different_param[k1.lower()] = {
                        "lobsterin1": v1,
                        "lobsterin2": other[k1_in_other],
                    }
            elif v1 != other[k1_lower]:
                different_param[k1.lower()] = {
                    "lobsterin1": v1,
                    "lobsterin2": other[k1_in_other],
                }
            else:
                similar_param[k1.lower()] = v1

        for k2, v2 in other.items():
            if (
                k2.lower() not in similar_param
                and k2.lower() not in different_param
                and k2.lower() not in [key.lower() for key in self]
            ):
                different_param[k2.lower()] = {"lobsterin1": None, "lobsterin2": v2}
        return {"Same": similar_param, "Different": different_param}

    def _get_nbands(self, structure: Structure):
        """Get number of bands."""
        if self.get("basisfunctions") is None:
            raise ValueError("No basis functions are provided. The program cannot calculate nbands.")

        basis_functions: list[str] = []
        for string_basis in self["basisfunctions"]:
            # string_basis.lstrip()
            string_basis_raw = string_basis.strip().split(" ")
            while "" in string_basis_raw:
                string_basis_raw.remove("")
            for _idx in range(int(structure.composition.element_composition[string_basis_raw[0]])):
                basis_functions.extend(string_basis_raw[1:])

        no_basis_functions = 0
        for basis in basis_functions:
            if "s" in basis:
                no_basis_functions += 1
            elif "p" in basis:
                no_basis_functions = no_basis_functions + 3
            elif "d" in basis:
                no_basis_functions = no_basis_functions + 5
            elif "f" in basis:
                no_basis_functions = no_basis_functions + 7

        return int(no_basis_functions)

    def write_lobsterin(self, path="lobsterin", overwritedict=None):
        """Write a lobsterin file.

        Args:
            path (str): filename of the lobsterin file that will be written
            overwritedict (dict): dict that can be used to overwrite lobsterin, e.g. {"skipdos": True}
        """
        # will overwrite previous entries
        # has to search first if entry is already in Lobsterindict (due to case insensitivity)
        if overwritedict is not None:
            for key, entry in overwritedict.items():
                self[key] = entry
                for key2 in self:
                    if key.lower() == key2.lower():
                        self[key2] = entry

        filename = path

        with open(filename, mode="w", encoding="utf-8") as file:
            for key in Lobsterin.AVAILABLE_KEYWORDS:
                if key.lower() in [element.lower() for element in self]:
                    if key.lower() in [element.lower() for element in Lobsterin.FLOAT_KEYWORDS]:
                        file.write(f"{key} {self.get(key)}\n")
                    elif key.lower() in [element.lower() for element in Lobsterin.BOOLEAN_KEYWORDS]:
                        # checks if entry is True or False
                        for key_here in self:
                            if key.lower() == key_here.lower():
                                file.write(f"{key}\n")
                    elif key.lower() in [element.lower() for element in Lobsterin.STRING_KEYWORDS]:
                        file.write(f"{key} {self.get(key)}\n")
                    elif key.lower() in [element.lower() for element in Lobsterin.LISTKEYWORDS]:
                        for entry in self.get(key):
                            file.write(f"{key} {entry}\n")

    def as_dict(self):
        """MSONable dict"""
        dct = dict(self)
        dct["@module"] = type(self).__module__
        dct["@class"] = type(self).__name__
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Lobsterin
        """
        return cls({k: v for k, v in dct.items() if k not in ["@module", "@class"]})

    def write_INCAR(
        self,
        incar_input: str = "INCAR",
        incar_output: str = "INCAR.lobster",
        poscar_input: str = "POSCAR",
        isym: int = -1,
        further_settings: dict | None = None,
    ):
        """Will only make the run static, insert nbands, make ISYM=-1, set LWAVE=True and write a new INCAR.
        You have to check for the rest.

        Args:
            incar_input (str): path to input INCAR
            incar_output (str): path to output INCAR
            poscar_input (str): path to input POSCAR
            isym (int): isym equal to -1 or 0 are possible. Current Lobster version only allow -1.
            further_settings (dict): A dict can be used to include further settings, e.g. {"ISMEAR":-5}
        """
        # reads old incar from file, this one will be modified
        incar = Incar.from_file(incar_input)
        warnings.warn("Please check your incar_input before using it. This method only changes three settings!")
        if isym == -1:
            incar["ISYM"] = -1
        elif isym == 0:
            incar["ISYM"] = 0
        else:
            raise ValueError(f"Got {isym=}, must be -1 or 0")
        incar["NSW"] = 0
        incar["LWAVE"] = True
        # get nbands from _get_nbands (use basis set that is inserted)
        incar["NBANDS"] = self._get_nbands(Structure.from_file(poscar_input))
        if further_settings is not None:
            for key, item in further_settings.items():
                incar[key] = item
        incar.write_file(incar_output)

    @staticmethod
    def get_basis(
        structure: Structure,
        potcar_symbols: list,
        address_basis_file: str | None = None,
    ):
        """Get the basis from given potcar_symbols (e.g., ["Fe_pv","Si"]

        Args:
            structure (Structure): Structure object
            potcar_symbols: list of potcar symbols

        Returns:
            returns basis
        """
        if address_basis_file is None:
            address_basis_file = f"{MODULE_DIR}/lobster_basis/BASIS_PBE_54_standard.yaml"
        potcar_names = list(potcar_symbols)

        atom_types_potcar = [name.split("_")[0] for name in potcar_names]

        if set(structure.symbol_set) != set(atom_types_potcar):
            raise ValueError("Your POSCAR does not correspond to your POTCAR!")
        basis = loadfn(address_basis_file)["BASIS"]

        basis_functions = []
        list_forin = []
        for idx, name in enumerate(potcar_names):
            if name not in basis:
                raise ValueError(
                    f"Missing basis information for POTCAR symbol: {name}. Please provide the basis manually."
                )
            basis_functions.append(basis[name].split())
            list_forin.append(f"{atom_types_potcar[idx]} {basis[name]}")
        return list_forin

    @staticmethod
    def get_all_possible_basis_functions(
        structure: Structure,
        potcar_symbols: list,
        address_basis_file_min: str | None = None,
        address_basis_file_max: str | None = None,
    ):
        """
        Args:
            structure: Structure object
            potcar_symbols: list of the potcar symbols
            address_basis_file_min: path to file with the minimum required basis by the POTCAR
            address_basis_file_max: path to file with the largest possible basis of the POTCAR.

        Returns:
            list[dict]: Can be used to create new Lobsterin objects in
                standard_calculations_from_vasp_files as dict_for_basis
        """
        max_basis = Lobsterin.get_basis(
            structure=structure,
            potcar_symbols=potcar_symbols,
            address_basis_file=address_basis_file_max or f"{MODULE_DIR}/lobster_basis/BASIS_PBE_54_max.yaml",
        )
        min_basis = Lobsterin.get_basis(
            structure=structure,
            potcar_symbols=potcar_symbols,
            address_basis_file=address_basis_file_min or f"{MODULE_DIR}/lobster_basis/BASIS_PBE_54_min.yaml",
        )
        all_basis = get_all_possible_basis_combinations(min_basis=min_basis, max_basis=max_basis)
        list_basis_dict = []
        for basis in all_basis:
            basis_dict = {}

            for elba in basis:
                basplit = elba.split()
                basis_dict[basplit[0]] = " ".join(basplit[1:])
            list_basis_dict.append(basis_dict)
        return list_basis_dict

    @staticmethod
    def write_POSCAR_with_standard_primitive(
        POSCAR_input="POSCAR", POSCAR_output="POSCAR.lobster", symprec: float = 0.01
    ):
        """Write a POSCAR with the standard primitive cell.
        This is needed to arrive at the correct kpath.

        Args:
            POSCAR_input (str): filename of input POSCAR
            POSCAR_output (str): filename of output POSCAR
            symprec (float): precision to find symmetry
        """
        structure = Structure.from_file(POSCAR_input)
        kpath = HighSymmKpath(structure, symprec=symprec)
        new_structure = kpath.prim
        new_structure.to(fmt="POSCAR", filename=POSCAR_output)

    @staticmethod
    def write_KPOINTS(
        POSCAR_input: str = "POSCAR",
        KPOINTS_output="KPOINTS.lobster",
        reciprocal_density: int = 100,
        isym: int = -1,
        from_grid: bool = False,
        input_grid: Tuple3Ints = (5, 5, 5),
        line_mode: bool = True,
        kpoints_line_density: int = 20,
        symprec: float = 0.01,
    ):
        """Write a KPOINT file for lobster (only ISYM=-1 and ISYM=0 are possible), grids are Gamma-centered.

        Args:
            POSCAR_input (str): path to POSCAR
            KPOINTS_output (str): path to output KPOINTS
            reciprocal_density (int): Grid density
            isym (int): either -1 or 0. Current Lobster versions only allow -1.
            from_grid (bool): If True KPOINTS will be generated with the help of a grid given in input_grid.
                Otherwise, they will be generated from the reciprocal_density
            input_grid (tuple): grid to generate the KPOINTS file
            line_mode (bool): If True, band structure will be generated
            kpoints_line_density (int): density of the lines in the band structure
            symprec (float): precision to determine symmetry
        """
        structure = Structure.from_file(POSCAR_input)
        if not from_grid:
            kpoint_grid = Kpoints.automatic_density_by_vol(structure, reciprocal_density).kpts
            mesh = kpoint_grid[0]
        else:
            mesh = input_grid

        # The following code is taken from: SpacegroupAnalyzer
        # we need to switch off symmetry here
        matrix = structure.lattice.matrix
        positions = structure.frac_coords
        unique_species: list[Composition] = []
        zs = []
        magmoms = []

        for species, group in itertools.groupby(structure, key=lambda s: s.species):
            if species in unique_species:
                ind = unique_species.index(species)
                zs.extend([ind + 1] * len(tuple(group)))
            else:
                unique_species.append(species)
                zs.extend([len(unique_species)] * len(tuple(group)))

        for site in structure:
            if hasattr(site, "magmom"):
                magmoms.append(site.magmom)
            elif site.is_ordered and hasattr(site.specie, "spin"):
                magmoms.append(site.specie.spin)
            else:
                magmoms.append(0)

        # For now, we are setting magmom to zero. (Taken from INCAR class)
        cell = matrix, positions, zs, magmoms
        # TODO: what about this shift?
        mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])

        # get the kpoints for the grid
        if isym == -1:
            kpts = []
            weights = []
            all_labels = []
            for gp in grid:
                kpts.append(gp.astype(float) / mesh)
                weights.append(float(1))
                all_labels.append("")
        elif isym == 0:
            # time reversal symmetry: k and -k are equivalent
            kpts = []
            weights = []
            all_labels = []
            newlist = [list(gp) for gp in list(grid)]
            mapping = []
            for gp in newlist:
                minus_gp = [-k for k in gp]
                if minus_gp in newlist and minus_gp != [0, 0, 0]:
                    mapping.append(newlist.index(minus_gp))
                else:
                    mapping.append(newlist.index(gp))

            for igp, gp in enumerate(newlist):
                if mapping[igp] > igp:
                    kpts.append(np.array(gp).astype(float) / mesh)
                    weights.append(float(2))
                    all_labels.append("")
                elif mapping[igp] == igp:
                    kpts.append(np.array(gp).astype(float) / mesh)
                    weights.append(float(1))
                    all_labels.append("")

        else:
            raise ValueError(f"Got {isym=}, must be -1 or 0")
        # line mode
        if line_mode:
            kpath = HighSymmKpath(structure, symprec=symprec)
            if not np.allclose(kpath.prim.lattice.matrix, structure.lattice.matrix):
                raise ValueError(
                    "You are not using the standard primitive cell. The k-path is not correct. Please generate a "
                    "standard primitive cell first."
                )

            frac_k_points, labels = kpath.get_kpoints(line_density=kpoints_line_density, coords_are_cartesian=False)

            for k, f in enumerate(frac_k_points):
                kpts.append(f)
                weights.append(0.0)
                all_labels.append(labels[k])
        comment = f"{isym=}, grid: {mesh} plus kpoint path" if line_mode else f"{isym=}, grid: {mesh}"

        kpoint_object = Kpoints(
            comment=comment,
            style=Kpoints.supported_modes.Reciprocal,
            num_kpts=len(kpts),
            kpts=tuple(kpts),
            kpts_weights=weights,
            labels=all_labels,
        )

        kpoint_object.write_file(filename=KPOINTS_output)

    @classmethod
    def from_file(cls, lobsterin: str) -> Self:
        """
        Args:
            lobsterin (str): path to lobsterin.

        Returns:
            Lobsterin object
        """
        with zopen(lobsterin, mode="rt") as file:
            data = file.read().split("\n")
        if len(data) == 0:
            raise RuntimeError("lobsterin file contains no data.")
        lobsterin_dict: dict[str, Any] = {}

        for datum in data:
            if datum.startswith(("!", "#", "//")):
                continue  # ignore comments
            pattern = r"\b[^!#//]+"  # exclude comments after commands
            if matched_pattern := re.findall(pattern, datum):
                raw_datum = matched_pattern[0].replace("\t", " ")  # handle tab in between and end of command
                key_word = raw_datum.strip().split(" ")  # extract keyword
                key = key_word[0].lower()
                if len(key_word) > 1:
                    # check which type of keyword this is, handle accordingly
                    if key not in [datum2.lower() for datum2 in Lobsterin.LISTKEYWORDS]:
                        if key not in [datum2.lower() for datum2 in Lobsterin.FLOAT_KEYWORDS]:
                            if key in lobsterin_dict:
                                raise ValueError(f"Same keyword {key} twice!")
                            lobsterin_dict[key] = " ".join(key_word[1:])
                        elif key in lobsterin_dict:
                            raise ValueError(f"Same keyword {key} twice!")
                        else:
                            lobsterin_dict[key] = float("nan" if key_word[1].strip() == "None" else key_word[1])
                    elif key not in lobsterin_dict:
                        lobsterin_dict[key] = [" ".join(key_word[1:])]
                    else:
                        lobsterin_dict[key].append(" ".join(key_word[1:]))
                elif len(key_word) > 0:
                    lobsterin_dict[key] = True

        return cls(lobsterin_dict)

    @staticmethod
    def _get_potcar_symbols(POTCAR_input: str) -> list:
        """
        Will return the name of the species in the POTCAR.

        Args:
            POTCAR_input (str): string to potcar file

        Returns:
            list of the names of the species in string format
        """
        potcar = Potcar.from_file(POTCAR_input)
        for pot in potcar:
            if pot.potential_type != "PAW":
                raise ValueError("Lobster only works with PAW! Use different POTCARs")

        # Warning about a bug in lobster-4.1.0
        with zopen(POTCAR_input, mode="r") as file:
            data = file.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if "SHA256" in data or "COPYR" in data:
            warnings.warn(
                "These POTCARs are not compatible with "
                "Lobster up to version 4.1.0."
                "\n The keywords SHA256 and COPYR "
                "cannot be handled by Lobster"
                " \n and will lead to wrong results."
            )

        if potcar.functional != "PBE":
            raise RuntimeError("We only have BASIS options for PBE so far")

        return [name["symbol"] for name in potcar.spec]

    @classmethod
    def standard_calculations_from_vasp_files(
        cls,
        POSCAR_input: str = "POSCAR",
        INCAR_input: str = "INCAR",
        POTCAR_input: str | None = None,
        Vasprun_output: str = "vasprun.xml",
        dict_for_basis: dict | None = None,
        option: str = "standard",
    ):
        """
        Will generate Lobsterin with standard settings.

        Args:
            POSCAR_input (str): path to POSCAR
            INCAR_input (str): path to INCAR
            POTCAR_input (str): path to POTCAR
            dict_for_basis (dict): can be provided: it should look the following:
                dict_for_basis={"Fe":'3p 3d 4s 4f', "C": '2s 2p'} and will overwrite all settings from POTCAR_input

            option (str): 'standard' will start a normal lobster run where COHPs, COOPs, DOS, CHARGE etc. will be
                calculated
                'standard_with_energy_range_from_vasprun' will start a normal lobster run for entire energy range
                of VASP static run. vasprun.xml file needs to be in current directory.
                'standard_from_projection' will start a normal lobster run from a projection
                'standard_with_fatband' will do a fatband calculation, run over all orbitals
                'onlyprojection' will only do a projection
                'onlydos' will only calculate a projected dos
                'onlycohp' will only calculate cohp
                'onlycoop' will only calculate coop
                'onlycohpcoop' will only calculate cohp and coop

        Returns:
            Lobsterin Object with standard settings
        """
        warnings.warn(
            "Always check and test the provided basis functions. The spilling of your Lobster calculation might help"
        )
        # warn that fatband calc cannot be done with tetrahedron method at the moment
        if option not in [
            "standard",
            "standard_from_projection",
            "standard_with_fatband",
            "standard_with_energy_range_from_vasprun",
            "onlyprojection",
            "onlydos",
            "onlycohp",
            "onlycoop",
            "onlycobi",
            "onlycohpcoop",
            "onlycohpcoopcobi",
            "onlymadelung",
        ]:
            raise ValueError("The option is not valid!")

        lobsterin_dict: dict[str, Any] = {
            # this basis set covers most elements
            "basisSet": "pbeVaspFit2015",
            # energies around e-fermi
            "COHPstartEnergy": -35.0,
            "COHPendEnergy": 5.0,
        }

        if option in {
            "standard",
            "standard_with_energy_range_from_vasprun",
            "onlycohp",
            "onlycoop",
            "onlycobi",
            "onlycohpcoop",
            "onlycohpcoopcobi",
            "standard_with_fatband",
        }:
            # every interaction with a distance of 6.0 is checked
            lobsterin_dict["cohpGenerator"] = "from 0.1 to 6.0 orbitalwise"
            # the projection is saved
            lobsterin_dict["saveProjectionToFile"] = True

        if option == "standard_from_projection":
            lobsterin_dict["cohpGenerator"] = "from 0.1 to 6.0 orbitalwise"
            lobsterin_dict["loadProjectionFromFile"] = True

        if option == "standard_with_energy_range_from_vasprun":
            vasp_run = Vasprun(Vasprun_output)
            lobsterin_dict["COHPstartEnergy"] = round(
                min(vasp_run.complete_dos.energies - vasp_run.complete_dos.efermi), 4
            )
            lobsterin_dict["COHPendEnergy"] = round(
                max(vasp_run.complete_dos.energies - vasp_run.complete_dos.efermi), 4
            )
            lobsterin_dict["COHPSteps"] = len(vasp_run.complete_dos.energies)

        # TODO: add cobi here! might be relevant lobster version
        if option == "onlycohp":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipcoop"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlycoop":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipcohp"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlycohpcoop":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlycohpcoopcobi":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlydos":
            lobsterin_dict["skipcohp"] = True
            lobsterin_dict["skipcoop"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlyprojection":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipcohp"] = True
            lobsterin_dict["skipcoop"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            lobsterin_dict["saveProjectionToFile"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlycobi":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipcohp"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True
            lobsterin_dict["skipMadelungEnergy"] = True

        if option == "onlymadelung":
            lobsterin_dict["skipdos"] = True
            lobsterin_dict["skipcohp"] = True
            lobsterin_dict["skipcoop"] = True
            lobsterin_dict["skipPopulationAnalysis"] = True
            lobsterin_dict["skipGrossPopulation"] = True
            lobsterin_dict["saveProjectionToFile"] = True
            # lobster-4.1.0
            lobsterin_dict["skipcobi"] = True

        incar = Incar.from_file(INCAR_input)
        if incar["ISMEAR"] == 0:
            lobsterin_dict["gaussianSmearingWidth"] = incar["SIGMA"]
        if incar["ISMEAR"] != 0 and option == "standard_with_fatband":
            raise ValueError("ISMEAR has to be 0 for a fatband calculation with Lobster")
        if dict_for_basis is not None:
            # dict_for_basis={"Fe":'3p 3d 4s 4f', "C": '2s 2p'}
            # will just insert this basis and not check with poscar
            basis = [f"{key} {value}" for key, value in dict_for_basis.items()]
        elif POTCAR_input is not None:
            # get basis from POTCAR
            potcar_names = Lobsterin._get_potcar_symbols(POTCAR_input=POTCAR_input)

            basis = Lobsterin.get_basis(structure=Structure.from_file(POSCAR_input), potcar_symbols=potcar_names)
        else:
            raise ValueError("basis cannot be generated")
        lobsterin_dict["basisfunctions"] = basis
        if option == "standard_with_fatband":
            lobsterin_dict["createFatband"] = basis

        return cls(lobsterin_dict)


def get_all_possible_basis_combinations(min_basis: list, max_basis: list) -> list:
    """
    Args:
        min_basis: list of basis entries: e.g. ['Si 3p 3s ']
        max_basis: list of basis entries: e.g. ['Si 3p 3s '].

    Returns:
        list[list[str]]: all possible combinations of basis functions, e.g. [['Si 3p 3s']]
    """
    max_basis_lists = [x.split() for x in max_basis]
    min_basis_lists = [x.split() for x in min_basis]

    # get all possible basis functions
    basis_dict: dict[str, dict] = {}
    for iel, el in enumerate(max_basis_lists):
        basis_dict[el[0]] = {"fixed": [], "variable": [], "combinations": []}
        for basis in el[1:]:
            if basis in min_basis_lists[iel]:
                basis_dict[el[0]]["fixed"].append(basis)
            if basis not in min_basis_lists[iel]:
                basis_dict[el[0]]["variable"].append(basis)
        for L in range(len(basis_dict[el[0]]["variable"]) + 1):
            for subset in itertools.combinations(basis_dict[el[0]]["variable"], L):
                basis_dict[el[0]]["combinations"].append(" ".join([el[0]] + basis_dict[el[0]]["fixed"] + list(subset)))

    list_basis = [item["combinations"] for item in basis_dict.values()]

    # get all combinations
    start_basis = list_basis[0]
    if len(list_basis) > 1:
        for el in list_basis[1:]:
            new_start_basis = []
            for elbasis in start_basis:
                for elbasis2 in el:
                    if not isinstance(elbasis, list):
                        new_start_basis.append([elbasis, elbasis2])
                    else:
                        new_start_basis.append([*elbasis.copy(), elbasis2])
            start_basis = new_start_basis
        return start_basis
    return [[basis] for basis in start_basis]
