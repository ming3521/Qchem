from pathlib import Path
import shutil
from openfermionpsi4 import run_psi4
from openfermion.chem import MolecularData

class Psi4Driver:
    def __init__(self, symbols, geometry, basis='sto-3g', multiplicity=1, charge=0):
        self.symbols = symbols
        self.geometry = geometry
        if len(symbols) != len(geometry):
            raise ValueError("symbols and geometry must have the same length")
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge


    def run(self, filename = None, run_fci=False):
        if filename is None:
            workdir = Path.cwd() / ".psi4_temp_data"
            if workdir.exists() and workdir.is_dir():
                shutil.rmtree(workdir)
            workdir.mkdir(exist_ok=True)
            path = (workdir / "temp_mol").as_posix()

        else:
            path = filename
        # 定义分子存储对象
        molecule = MolecularData(
            geometry=[(self.symbols[i], self.geometry[i]) for i in range(len(self.symbols))], 
            basis=self.basis, 
            multiplicity=self.multiplicity, 
            charge=self.charge,
            filename = path
        )

        try:
            molecule.load()
        except (FileNotFoundError, EOFError, Exception):
            molecule = run_psi4(molecule, run_scf=True, run_fci=run_fci, delete_input=True,delete_output=True)
        
        return molecule