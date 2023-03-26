import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data as gdata
import torch_geometric.nn as gnn
from rdkit.Chem import AllChem
from rdkit.Chem import BondType
from rdkit.Chem import PandasTools
from tqdm import tqdm
import torch_scatter
from torch.nn import Module, Sequential, Linear, ELU
from torch_geometric.data import DataLoader, Batch
from matplotlib import pyplot as plt
from chem_utils.rxn_template.template_apply import TemplateApply
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(42) # Setting the seed
# 上一节生成的反应和反应模板数据
rxn_data_file_path = "data/rxn_and_template.tsv"
template_file_path = "data/template.tsv"

# 训练和测试数据的存储路径
validation_raw_data_file_path = "data/validation_raw_data.tsv"
train_raw_data_file_path = "data/train_raw_data.tsv"
test_raw_data_file_path = "data/test_raw_data.tsv"
validation_processed_data_dir_path = "data/processed_validation"
train_processed_data_dir_path = "data/processed_train"
test_processed_data_dir_path = "data/processed_test"
template_df = pd.read_csv(template_file_path, sep='\t', encoding='utf-8')


def template_code_to_one_hot_idx(template_code: str):
    query = template_df.query(f"template_code=='{template_code}'")
    if len(query) == 0:
        raise ValueError(f"template.tsv中没有这个template_code: {template_code}")
    return query.index[0]
    # 测试一下，通过反应模板的code，获得了类别序号
template_code_to_one_hot_idx('T120')
def one_hot_idx_to_template_smarts(one_hot_idx: int):
    return template_df.loc[one_hot_idx, "template_smarts"]
    # 测试一下，通过类别序号，得到了反应模板的smarts
one_hot_idx_to_template_smarts(120)





# 载入反应数据

rxn_df = pd.read_csv(rxn_data_file_path, sep='\t', encoding='utf-8')

# 打乱数据
rxn_df = rxn_df.sample(frac=1)

#设置训练集; 验证集;测试集 = 8:1: 1
train_num = int(len(rxn_df)*0.8)
test_validation=int(len(rxn_df)*0.1)
train_rxn_df = rxn_df.iloc[:train_num, ]
validation_rxn_df = rxn_df.iloc[train_num:(test_validation+train_num), ]
test_rxn_df = rxn_df.iloc[(test_validation+train_num):, ]
train_rxn_df.to_csv(train_raw_data_file_path, sep='\t', encoding='utf-8', index=False)
validation_rxn_df.to_csv(validation_raw_data_file_path, sep='\t', encoding='utf-8', index=False)
test_rxn_df.to_csv(test_raw_data_file_path, sep='\t', encoding='utf-8', index=False)







# ===== Atom =====
# 通过下列方法，获得原子特征

def get_atom_features(atom):
    return [
            # int(atom.GetAtomicNum()==6),
            # int(atom.GetAtomicNum()==7),
            # int(atom.GetAtomicNum()==8),
            # int(atom.GetAtomicNum()==9),
            # int(atom.GetAtomicNum()==15),
            # int(atom.GetAtomicNum()==16),
            # int(atom.GetAtomicNum()==17),
            # int(atom.GetAtomicNum()==35),
            # int(atom.GetAtomicNum()==53),
            # int(str(atom.GetHybridization())=="SP"),
            # int(str(atom.GetHybridization())=="SP2"),
            # int(str(atom.GetHybridization())=="SP3"),
            atom.GetAtomicNum(),
            atom.GetHybridization(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            # atom.GetDoubleProp('_GasteigerCharge'),
            # atom.GetFormalCharge(),
            # 0,

            atom.GetTotalValence(),]

def get_atoms_features(mol):
    res = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        # print(get_atom_features(atom))
        # print("\n")
        res.append(get_atom_features(atom))
    return res
        




# ===== Bond =====
# 通过下列方法，获得键的特征及分子中的键连关系

def get_bond_features(bond):
    res = [ 
        bond.GetBondType() == BondType.SINGLE,
        bond.GetBondType() == BondType.DOUBLE,
        bond.GetBondType() == BondType.TRIPLE,
        bond.GetBondType() == BondType.AROMATIC,
        bond.GetIsConjugated(),    
        bond.IsInRing()
            ]
    # res = [0]*5
    # if bond.GetBondType() == BondType.SINGLE:
    #     idx = 0
    # elif bond.GetBondType() == BondType.DOUBLE:
    #     idx = 1
    # elif bond.GetBondType() == BondType.TRIPLE:
    #     idx= 2
    # elif bond.GetBondType() == BondType.AROMATIC:
    #     idx = 3
    # else:
    #     idx = 4
    # res[idx] = 1
    return res

def get_bonds_features_and_connections(mol):
    bonds_features = []
    connections = []
    for bond in mol.GetBonds():
        bonds_features.append(get_bond_features(bond))
        bonds_features.append(get_bond_features(bond))
        connections.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        connections.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    return bonds_features, connections





# ===== Mol =====
# 获得分子中的原子特征、键特征、键连关系，并存储于Data对象中

def get_gnn_data_from_smiles(smiles: str):
    mol = AllChem.MolFromSmiles(smiles)
    AllChem.ComputeGasteigerCharges(mol)
    atoms_features = get_atoms_features(mol)
    bonds_features, connections = get_bonds_features_and_connections(mol)
    data = gdata.Data()
    data.x = torch.tensor(atoms_features, dtype=torch.float)
    data.edge_attr = torch.tensor(bonds_features, dtype=torch.float)
    data.edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
    return data
class RetroSynPreDataset(gdata.InMemoryDataset):
    
    def __init__(self, train_or_test, raw_data_file_path: str, processed_data_dir_path: str):
        """构造方法：用于初始化新创建对象的状态
        :param train_or_test: 'train'或者'test'
        :param raw_data_file_path: 原始数据文件的路径
        :param processed_data_dir_path: 处理后的数据的存储文件夹路径
        
        """
        self._raw_data_fp = raw_data_file_path
        self._train_or_test = train_or_test
        root = processed_data_dir_path
        
        super(RetroSynPreDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        """ 没用，但因为父对象的要求，此处必须定义。父对象就是 gdata.InMemoryDataset
        """
        return [] 
    
        
    def download(self):
        """ 没用，但因为父对象的要求，此处必须定义
        """
        pass
    
    @property
    def processed_file_names(self):
        """ 
        :return: 返回处理后的数据的文件名列表，这里我们只写了一个文件名，因为待会儿我们会把处理后的数据全存到一个文件中
        
        父对象初始化后会检查processed_file_names中的文件是否存在，如果不存在，会调用process方法，就是下一个方法。
        """
        return [f"retro_synpre_{self._train_or_test}.rxns"]
    
    def process(self):
        """ 在这个方法中会处理数据，并存储。
        """
        # 加载原始数据
        rxn_df = pd.read_csv(self._raw_data_fp, sep='\t', encoding='utf-8')
        data_list = []
        with tqdm(total=len(rxn_df))as pbar:
            pbar.set_description_str(f"{self._train_or_test} process data")
            for _, row in rxn_df.iterrows():
                # 遍历原始数据，依次获得产物，及对应的模板code
                rxn_smiles = row.rxn_smiles
                product_smiles = rxn_smiles.split('>')[-1]
                
                template_one_hot_idx = template_code_to_one_hot_idx(row.template_code)
                template_one_hot_vec = [0]*len(template_df)
                template_one_hot_vec[template_one_hot_idx] = 1
                
                # 转化为Data对象
                data = get_gnn_data_from_smiles(product_smiles)
                data.y = torch.tensor([template_one_hot_idx], dtype=torch.long)
                
                data_list.append(data)
                pbar.update(1)
        data, slices = self.collate(data_list)
        
        # 存储下来
        torch.save((data, slices), self.processed_paths[0])
# 这里，我们对于训练集和测试集，分别获得他们的RetroSynPreDataset对象。
# 如果这是你第一次运行这里，会看到process的进度条
validation_dataset = RetroSynPreDataset("validation", validation_raw_data_file_path, validation_processed_data_dir_path)
train_dataset = RetroSynPreDataset("train", train_raw_data_file_path, train_processed_data_dir_path)
test_dataset = RetroSynPreDataset("test", test_raw_data_file_path, test_processed_data_dir_path)



# 边的更新
class EdgeModel(Module):
    def __init__(self, num_node_features, num_edge_features, out_features):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(Linear(num_node_features + num_node_features + num_edge_features, 32),
                                   ELU(),
                                   Linear(32, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)
    
    
# 节点更新
class NodeModel(Module):
    def __init__(self, num_node_features, num_edge_features_out, out_features):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(Linear(num_node_features + num_edge_features_out, 64),
                                     ELU(),
                                     Linear(64, 64))
        self.node_mlp_2 = Sequential(Linear(num_node_features + 64, 64),
                                     ELU(),
                                     Linear(64, out_features))
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = torch_scatter.scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
        

# 全局更新
class GlobalModel(Module):
    def __init__(self, num_node_features, num_global_features, out_channels):
        super(GlobalModel, self).__init__()
        self.global_mlp = Sequential(Linear(num_global_features + num_node_features, 64),
                                     ELU(),
                                     Linear(64, out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        if u is None:
            out = torch_scatter.scatter_mean(x, batch, dim=0)
        else:
            out = torch.cat([u, torch_scatter.scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)

    
# GNN
class Net(Module):
    
    def __init__(self, num_node_features, num_edge_features, out_channels):
        """ 构造方法
        :param num_node_features: 节点特征的数目
        :param num_edge_features: 边的特征的数目
        :param out_channels: 反应模板的数目
        """
        super(Net, self).__init__()
        # 节点和边的特征进行标准化处理
        self.node_normal = gnn.BatchNorm(num_node_features)
        self.edge_normal = gnn.BatchNorm(num_edge_features)
        
        # 可以看到，每层GNN都是依次进行Edge更新，Node更新，Global更新
        self.meta1 = gnn.MetaLayer(EdgeModel(num_node_features, num_edge_features, 256),
                                   NodeModel(num_node_features, 256, 32),
                                   GlobalModel(32, 0, 32))
        self.meta2 = gnn.MetaLayer(EdgeModel(32, 256, 256),
                                   NodeModel(32, 256, 32),
                                   GlobalModel(32, 32, 32))
        self.meta3 = gnn.MetaLayer(EdgeModel(32, 256, 256),
                                   NodeModel(32, 256, 32),
                                   GlobalModel(32, 32, 32))
        self.meta4 = gnn.MetaLayer(EdgeModel(32, 256, 256),
                                   NodeModel(32, 256, 32),
                                   GlobalModel(32, 32, 32))
        self.meta5 = gnn.MetaLayer(EdgeModel(32, 256, 256),
        NodeModel(32, 256, 32),
                                   GlobalModel(32, 32, 32))
        self.meta6 = gnn.MetaLayer(EdgeModel(32, 256, 256),
                                   NodeModel(32, 256, 32),
                                   GlobalModel(32, 32, 32))
        self.meta7 = gnn.MetaLayer(EdgeModel(32, 256, 32),
                                   NodeModel(32, 32, 32),
                                   GlobalModel(32, 32, 64))
        self.lin1 = Linear(64, 256)
        self.lin2 = Linear(256, out_channels)
        #         self.meta1 = gnn.MetaLayer(EdgeModel(num_node_features, num_edge_features, 128),
        #                            NodeModel(num_node_features, 128, 32),
        #                            GlobalModel(32, 0, 32))
        # self.meta2 = gnn.MetaLayer(EdgeModel(32, 128, 128),
        #                            NodeModel(32, 128, 32),
        #                            GlobalModel(32, 32, 32))
        # self.meta3 = gnn.MetaLayer(EdgeModel(32, 128, 128),
        #                            NodeModel(32, 128, 32),
        #                            GlobalModel(32, 32, 32))
        # self.meta4 = gnn.MetaLayer(EdgeModel(32, 128, 32),
        #                            NodeModel(32, 32, 32),
        #                            GlobalModel(32, 32, 64))
        # self.lin1 = Linear(64, 128)
        # self.lin2 = Linear(128, out_channels)
    def forward(self, data):
        x, edge_index, e, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_normal(x)
        e = self.edge_normal(e)

        x, e, g = self.meta1(x, edge_index, e, None, batch)
        x, e, g = self.meta2(x, edge_index, e, g, batch)
        x, e, g = self.meta3(x, edge_index, e, g, batch)
        x, e, g = self.meta4(x, edge_index, e, g, batch)
        x, e, g = self.meta5(x, edge_index, e, g, batch)
        x, e, g = self.meta6(x, edge_index, e, g, batch)
        x, e, g = self.meta7(x, edge_index, e, g, batch)
        y = F.elu(self.lin1(g))
        return self.lin2(y)
# 如果cuda可用，就用cuda，否则就用cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
print(device)
# 初始化模型
model = Net(5, 6, len(template_df)).to(device)
# 机器学习每一次参数的更新所需要的损失函数并不是由单个数据获得的，而是由一批数据加权得到的，这一批数据的数量就是batch size。
batch_size = 200

# 训练集和测试集的Dataloader
train_dataloader = DataLoader(train_dataset, batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

# 所有的数据集，需要训练很多次，max_epoch次，为了节省时间，这里定义的30比较小
max_epoch = 40

# 优化器Adam，学习率lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 通过下面的方法，阶梯式下降学习率，每过10个epoch，学习率下降2/10
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.40)
# 了解即可
def train_model_with_logger(epochs, logging_dir='runs/our_experiment'):
    # Create TensorBoard logger
    writer = SummaryWriter(logging_dir)

    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        test_loss = 0.0
        for _, data_inputs in enumerate(train_dataloader):

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)


            ## Step 2: Run the model on the input data
            preds = model(data_inputs)

            ## Step 3: Calculate the loss
            loss = F.cross_entropy(preds, data_inputs.y)

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()

        # Add average loss to TensorBoard
        epoch_loss /= len(train_dataloader)
        
        for _, data_inputs in enumerate(validation_dataloader):
            data_inputs = data_inputs.to(device)
            pred = model(data_inputs)
            loss = F.cross_entropy(pred, data_inputs.y)

            test_loss += loss.item()
        test_loss /= len(train_dataloader)


        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)
        writer.add_scalar('TEST_loss',
                          test_loss ,
                          global_step = epoch + 1)

    writer.close()
def train(epoch):
    
    model.train()
    tot_loss = 0
    tot_correct = 0
    tot_mol = 0
    with tqdm(total=len(train_dataloader))as pbar:
        
        pbar.set_description_str(f"train {epoch}")
        for n, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            pred = model(batch_data)

            print(pred)

            loss = F.cross_entropy(pred, batch_data.y)
            loss.backward()
            optimizer.step()
            
            num_mols = batch_data.num_graphs
            tot_mol += num_mols
            tot_loss += (loss.item()*num_mols)
            tot_correct += pred.max(dim=1)[1].eq(batch_data.y).sum().item()
            
            pbar.set_postfix_str(f"loss: {tot_loss/tot_mol}, acc: {tot_correct/tot_mol}")
            pbar.update(1)
    return tot_loss/tot_mol, tot_correct/tot_mol

# 了解即可
def test(epoch,dataloader):
    model.eval()
    tot_loss = 0
    tot_correct = 0
    tot_mol = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader))as pbar:
            pbar.set_description_str(f"test {epoch}")
            for n, batch_data in enumerate(dataloader):
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                loss = F.cross_entropy(pred, batch_data.y)

                num_mols = batch_data.num_graphs
                tot_mol += num_mols
                tot_loss += (loss.item()*num_mols)
                tot_correct += pred.max(dim=1)[1].eq(batch_data.y).sum().item()

                pbar.set_postfix_str(f"loss: {tot_loss/tot_mol}, acc: {tot_correct/tot_mol}")
                pbar.update(1)
    return tot_loss/tot_mol, tot_correct/tot_mol
# for epoch in range(max_epoch):
#     train_loss, test_acc = train(epoch)
#     test_loss, test_acc = test(epoch,validation_dataloader)
#     scheduler.step()
#     print(f"test loss: {test_loss}, test acc: {test_acc}")
train_model_with_logger(max_epoch)
# 了解即可
# test_loss, test_acc = test(epoch,validation_dataloader)
# print(f"test loss: {test_loss}, test acc: {test_acc}")

def get_topk_acc(max_k, pred_props, real_idxes):
    num_tot = real_idxes.size()[0]
    _, pred_idxes_list = pred_props.topk(max_k, 1, True, True)
    pred_idxes_list_t = pred_idxes_list.t()
    k_and_acc = {}
    for k in range(1, max_k+1):
        correct = pred_idxes_list_t.eq(real_idxes.reshape((1, num_tot)).expand_as(pred_idxes_list.t()))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        k_and_acc[k] = correct_k[0].item() / num_tot
    return k_and_acc
# 了解即可
model.eval()
all_pred = None
all_real = None
with torch.no_grad():
    with tqdm(total=len(test_dataloader))as pbar:
        pbar.set_description_str(f"test {max_epoch}")
        for n, batch_data in enumerate(test_dataloader):
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            all_pred = pred.to('cpu') if all_pred is None else torch.cat((all_pred, pred.to('cpu')))
            all_real = batch_data.y.to('cpu') if all_real is None else torch.cat((all_real, batch_data.y.to('cpu')))

            pbar.update(1)


# 得到k与accuracy的关系
k_and_acc = get_topk_acc(50, all_pred, all_real)

# 画线
plt.plot(k_and_acc.keys(), k_and_acc.values())

# 画点
plt.scatter(k_and_acc.keys(), k_and_acc.values(), s=10)

# x轴标题
plt.xlabel("K")

# y轴标题
plt.ylabel("Accuracy")

# 展示
plt.show()
print(k_and_acc.values())