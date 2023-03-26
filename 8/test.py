from rdkit.Chem import AllChem
from rdkit.Chem import Draw

# 解析SMILES
mol = AllChem.MolFromSmiles("C1=C(O)NC=C1")

# 生成INCHI
inchi = AllChem.MolToInchi(mol)
print(inchi)
# 此处使用pandas只是为了更美观地展示表格
import pandas as pd
atoms_df = pd.DataFrame(columns=["atom id", "symbol", "atomic number", "number of H", "formal charge", "valence", "hybridization", "is aromatic"])

for atom in mol.GetAtoms():
    
    aid = atom.GetIdx()
    
    # 符号
    symbol = atom.GetSymbol()
    
    # 元素序号
    atomic_num = atom.GetAtomicNum()
    
    # 连接的氢原子数，！！！特别注意：RDKit中不将H当作原子处理，而是当作其它原子的属性
    num_hs = atom.GetTotalNumHs()
    
    # 形式电荷
    charge = atom.GetFormalCharge()
    
    # 价态
    valence = atom.GetTotalValence()
    
    # 杂化类型
    hybrid = atom.GetHybridization()
    print(hybrid.values)
    # 芳香性
    is_aromatic = atom.GetDegree()
    
    atoms_df = atoms_df.append({"atom id": aid, "symbol": symbol, "atomic number": atomic_num, 
                                "number of H": num_hs, "formal charge": charge, "valence": valence, 
                                "hybridization": hybrid, "is aromatic": is_aromatic}, ignore_index=True)
print(atoms_df)