from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# 这是抽取反应模板要用的化学反应
rxn_smiles = "Cl[c:10]1[c:9]2[c:4]([n:3][c:2]([Cl:1])[n:11]1)[cH:5][cH:6][cH:7][cH:8]2.[CH3:13][NH2:14]>>[Cl:1][c:2]1[n:3][c:4]2[cH:5][cH:6][cH:7][cH:8][c:9]2[c:10]([NH:14][CH3:13])[n:11]1"
rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
Draw.ReactionToImage(rxn)
# 调用 chem_utils
from chem_utils.rxn_template.template_extractor import TemplateExtractor

# 抽取反应模板
template_smarts = TemplateExtractor.rxn_smiles_to_rxn_temp_smarts(rxn_smiles)
print(template_smarts)

# 反应模板在形式上也是一种反应
template = AllChem.ReactionFromSmarts(template_smarts)

# 画图
Draw.ReactionToImage(template)
import pandas as pd
rxn_df = pd.read_csv('data/sample_rxns.tsv', sep='\t', encoding='utf-8')
print(rxn_df)
from tqdm import tqdm

# 创建一个表格用于存储反应模版，包括三列：
# 1. template code: 模版编号
# 2. template smarts: 模版
# 3. num rxns covered: 模版覆盖的反应的数目
template_df = pd.DataFrame(columns=['template_code', 'template_smarts', 'num_rxns_covered'])

# 反应数据表格中添加一个新的列：template code
rxn_df['template_code'] = None

# 绘制进度条
with tqdm(total=len(rxn_df))as pbar:
    
    # 遍历反应数据
    for rxn_index, rxn_row in rxn_df.iterrows():
        rxn_smiles = rxn_row.rxn_smiles
        
        # *** 通过反应的smile，获得反应模版
        template_smarts = TemplateExtractor.rxn_smiles_to_rxn_temp_smarts(rxn_smiles)
        
        # 查询，模版表格中是否已经存在这个模版
        query_df = template_df.query(f"template_smarts=='{template_smarts}'")
        
        # 如果存在，则把对应模版的num_rxns_covered + 1
        if len(query_df) > 0:
            idx = query_df.index[0]
            num_rxns_covered = query_df.loc[idx, 'num_rxns_covered']
            template_code = query_df.loc[idx, 'template_code']
            template_df.loc[idx, 'num_rxns_covered'] = num_rxns_covered + 1
            
        # 如果不存在，则将模版添加到模版表格中，并设置num_rxns_covered为1
        else:
            template_code = f"T{len(template_df)}"
            template_df = template_df.append({"template_code": template_code, 
                                              "template_smarts": template_smarts, 
                                              "num_rxns_covered": 1}, ignore_index=True)
            
        # 在反应表格中，填入每一个反应对应的模版的template_code
        rxn_df.loc[rxn_index, 'template_code'] = template_code
        pbar.update(1)
        # 存储到 data/template.tsv
template_df.to_csv('data/template.tsv', sep='\t', encoding='utf-8', index=False)
template_df
# 存储到 data/rxn_and_template.tsv
rxn_df.to_csv('data/rxn_and_template.tsv', sep='\t', encoding='utf-8', index=False)
rxn_df