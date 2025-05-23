"""Pymatgen package configuration."""

from __future__ import annotations

import platform
import sys

import numpy as np
from setuptools import Extension, find_namespace_packages, setup

is_win_64 = sys.platform.startswith("win") and platform.machine().endswith("64")
extra_link_args = ["-Wl,--allow-multiple-definition"] if is_win_64 else []

with open("README.md") as file:
    long_description = file.read()

# unlike GitHub readmes, PyPI doesn't support <picture> tags used for responsive images
# (i.e. adaptive to OS light/dark mode)
# NOTE this manual fix won't work once we migrate to pyproject.toml
logo_url = "https://raw.githubusercontent.com/materialsproject/pymatgen/master/docs/_images/pymatgen.svg"
long_description = (
    f"<h1 align='center'><img alt='Logo' src='{logo_url}' height='70'></h1>" + long_description.split("</picture>")[-1]
)

setup(
    name="pymatgen",
    packages=find_namespace_packages(include=["pymatgen.*", "pymatgen.**.*"]),
    version="2024.6.4",
    python_requires=">=3.9",
    install_requires=[
        "matplotlib>=3.8",
        "monty>=2024.5.24",
        "networkx>=2.2",
        "numpy>=1.25.0",
        "palettable>=3.1.1",
        "pandas>=2",
        "plotly>=4.5.0",
        "pybtex>=0.24.0",
        "requests>=2.32",
        "ruamel.yaml>=0.17.0",
        "scipy>=1.13.0",
        "spglib>=2.0.2",
        "sympy>=1.2",
        "tabulate>=0.9",
        "tqdm>=4.60",
        "uncertainties>=3.1.4",
        "joblib>=1",
    ],
    extras_require={
        "ase": ["ase>=3.23.0"],
        # don't depend on tblite above 3.11 since unsupported https://github.com/tblite/tblite/issues/175
        "tblite": ["tblite[ase]>=0.3.0; python_version<'3.12'"],
        "vis": ["vtk>=6.0.0"],
        "abinit": ["netcdf4>=1.6.5"],
        "relaxation": ["matgl>=1.1.1", "chgnet>=0.3.8"],
        "electronic_structure": ["fdint>=2.0.2"],
        "dev": [
            "mypy>=1.10.0",
            "pre-commit>=3",
            "pytest-cov>=4",
            "pytest-split>=0.8",
            "pytest>8",
            "ruff>=0.4",
        ],
        "ci": [
            "pytest>=8",
            "pytest-cov>=4",
            "pytest-split>=0.8",
        ],
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "doc2dash",
        ],
        "optional": [
            "ase>=3.23.0",
            # TODO restore BoltzTraP2 when install fixed, hopefully following merge of
            # https://gitlab.com/sousaw/BoltzTraP2/-/merge_requests/18
            # caused CI failure due to ModuleNotFoundError: No module named 'packaging'
            # "BoltzTraP2>=22.3.2; platform_system!='Windows'",
            "chemview>=0.6",
            "chgnet>=0.3.8",
            "f90nml>=1.1.2",
            "galore>=0.6.1",
            "h5py>=3.11.0",
            "jarvis-tools>=2020.7.14",
            "matgl>=1.1.1",
            "netCDF4>=1.6.5",
            "phonopy>=2.23",
            "seekpath>=2.0.1",
            # don't depend on tblite above 3.11 since unsupported https://github.com/tblite/tblite/issues/175
            "tblite[ase]>=0.3.0; platform_system=='Linux' and python_version<'3.12'",
            # "hiphive>=0.6",
            # "openbabel>=3.1.1; platform_system=='Linux'",
        ],
        "numba": ["numba>=0.55"],
    },
    # All package data has to be explicitly defined. Do not use automated codes like last time. It adds
    # all sorts of useless files like test files and is prone to path errors.
    package_data={
        "pymatgen.analysis": ["*.yaml", "*.json", "*.csv"],
        "pymatgen.analysis.chemenv": [
            "coordination_environments/coordination_geometries_files/*.json",
            "coordination_environments/coordination_geometries_files/*.txt",
            "coordination_environments/strategy_files/ImprovedConfidenceCutoffDefaultParameters.json",
        ],
        "pymatgen.analysis.structure_prediction": ["*.yaml", "data/*.json"],
        "pymatgen.analysis.diffraction": ["*.json"],
        "pymatgen.analysis.magnetism": ["default_magmoms.yaml"],
        "pymatgen.analysis.solar": ["am1.5G.dat"],
        "pymatgen.entries": ["*.json.gz", "*.yaml", "data/*.json"],
        "pymatgen.core": ["*.json"],
        "pymatgen": ["py.typed"],
        "pymatgen.io.vasp": ["*.yaml", "*.json", "*.json.gz", "*.json.bz2"],
        "pymatgen.io.feff": ["*.yaml"],
        "pymatgen.io.cp2k": ["*.yaml"],
        "pymatgen.io.lobster": ["lobster_basis/*.yaml"],
        "pymatgen.command_line": ["*"],
        "pymatgen.util": ["structures/*.json", "*.json"],
        "pymatgen.vis": ["*.yaml"],
        "pymatgen.io.lammps": ["CoeffsDataType.yaml", "templates/*.template"],
        "pymatgen.symmetry": ["*.yaml", "*.json", "*.sqlite"],
    },
    author="Pymatgen Development Team",
    author_email="ongsp@ucsd.edu",
    maintainer="Shyue Ping Ong, Matthew Horton, Janosh Riebesell",
    maintainer_email="ongsp@ucsd.edu, mkhorton@lbl.gov, janosh.riebesell@gmail.com",
    url="https://pymatgen.org",
    license="MIT",
    project_urls={
        "Docs": "https://pymatgen.org",
        "Package": "https://pypi.org/project/pymatgen",
        "Repo": "https://github.com/materialsproject/pymatgen",
    },
    description="Python Materials Genomics is a robust materials "
    "analysis code that defines core object representations for "
    "structures and molecules with support for many electronic "
    "structure codes. It is currently the core analysis code "
    "powering the Materials Project "
    "(https://materialsproject.org).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "ABINIT",
        "analysis",
        "crystal",
        "diagrams",
        "electronic",
        "gaussian",
        "materials",
        "nwchem",
        "phase",
        "project",
        "qchem",
        "science",
        "structure",
        "VASP",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    ext_modules=[
        Extension(
            "pymatgen.optimization.linear_assignment",
            ["pymatgen/optimization/linear_assignment.pyx"],
            extra_link_args=extra_link_args,
        ),
        Extension(
            "pymatgen.util.coord_cython",
            ["pymatgen/util/coord_cython.pyx"],
            extra_link_args=extra_link_args,
        ),
        Extension(
            "pymatgen.optimization.neighbors",
            ["pymatgen/optimization/neighbors.pyx"],
            extra_link_args=extra_link_args,
        ),
    ],
    entry_points={
        "console_scripts": [
            "pmg = pymatgen.cli.pmg:main",
            "feff_plot_cross_section = pymatgen.cli.feff_plot_cross_section:main",
            "feff_plot_dos = pymatgen.cli.feff_plot_dos:main",
            "get_environment = pymatgen.cli.get_environment:main",
        ]
    },
    include_dirs=[np.get_include()],
)
