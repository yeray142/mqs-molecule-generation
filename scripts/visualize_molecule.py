from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles(
    "Cc1csc(=NS(=O)(=O)c2ccc(N)cc2)[nH]1"
)

Draw.MolToFile(mol, "molecule.png", size=(800, 800))