from __future__ import annotations

from shutil import which

import pandas as pd
import pytest
from pytest import approx

from pymatgen.command_line.vampire_caller import VampireCaller
from pymatgen.core.structure import Structure
from pymatgen.util.testing import TEST_FILES_DIR, PymatgenTest

TEST_DIR = f"{TEST_FILES_DIR}/analysis/magnetic_orderings"


@pytest.mark.skipif(not which("vampire-serial"), reason="vampire executable not present")
class TestVampireCaller(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        cls.Mn3Al = pd.read_json(f"{TEST_DIR}/Mn3Al.json")

        cls.compounds = [cls.Mn3Al]

        cls.structure_inputs = []
        cls.energy_inputs = []
        for c in cls.compounds:
            ordered_structures = list(c["structure"])
            ordered_structures = [Structure.from_dict(d) for d in ordered_structures]
            epa = list(c["energy_per_atom"])
            energies = [e * len(s) for (e, s) in zip(epa, ordered_structures)]

            cls.structure_inputs.append(ordered_structures)
            cls.energy_inputs.append(energies)

    def test_vampire(self):
        for structs, energies in zip(self.structure_inputs, self.energy_inputs):
            settings = {"start_t": 0, "end_t": 500, "temp_increment": 50}
            vc = VampireCaller(
                structs,
                energies,
                mc_box_size=3.0,
                equil_timesteps=1000,  # 1000
                mc_timesteps=2000,  # 2000
                user_input_settings=settings,
            )

            assert approx(vc.output.critical_temp) == 400
