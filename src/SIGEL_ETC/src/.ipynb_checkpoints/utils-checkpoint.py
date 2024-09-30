import math
import SpaGCN as spg
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import anndata as ad
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import squidpy as sq

def find_resolution(adata, low, high, target_clusters, max_iter=10):
    iteration = 0
    while low < high and iteration < max_iter:
        mid = (low + high) / 2
        sc.tl.leiden(adata, resolution=mid)
        num_clusters = adata.obs['leiden'].nunique()
        
        print(f'Iteration {iteration}: Resolution {mid} -> {num_clusters} clusters')
        
        if num_clusters == target_clusters:
            return mid  # 找到精确匹配时返回当前resolution
        elif num_clusters < target_clusters:
            low = mid  # 需要更多的聚类，增加resolution
        else:
            high = mid  # 需要更少的聚类，减少resolution
        
        iteration += 1
    
    return (low + high) / 2  # 返回最接近的resolution值

def sequential_gen(model, ETC_GAN, all_gmat_v2m, all_gmat_m2v):
    device = 'cuda:1'
    dataset_m = np.array(list(all_gmat_m2v.values()))
    key_m = list(all_gmat_m2v.keys())
    file =  f'data/mEmb/10x_mEmb_matrix.dat'
    with open(file,'rb') as f:
        all_gene_exp_matrices = pickle.load(f)
    all_gmat_v = {k:all_gene_exp_matrices[k].todense()[25:60,40: ]for k in list(all_gene_exp_matrices.keys())}
    key_v = list(all_gmat_v.keys())
    dataset_v = np.array(list(all_gmat_v.values()))
        
    data_path = 'data/mEmb/sqf_mEmb_adata.h5ad'
    adata = sc.read_h5ad(data_path)
    adata.X=csr_matrix(adata.X)
    coor_raw = adata.obsm['spatial']
    coor = np.zeros(coor_raw.shape)
    max_normalized_x = coor_raw[:, 0].max()
    max_normalized_y = coor_raw[:, 1].max()
    min_normalized_x = coor_raw[:, 0].min()
    min_normalized_y = coor_raw[:, 1].min()
    coor[:, 0] = coor_raw[:, 0] - min_normalized_x
    coor[:, 1] = coor_raw[:, 1] - min_normalized_y
    coor = np.ceil(coor*40)

    all_genes = adata.var.index.values
    adata.obs['array_x']=np.ceil(coor[:, 0]).astype(int)
    adata.obs['array_y']=np.ceil(coor[:, 1]).astype(int)

    all_gene_exp_matrices = {}
    shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
    i = 0
    for gene in all_genes:
        g_matrix = np.zeros(shape=shape)

        g = adata[:,gene].X.todense().tolist()
        #c = adata.obsm['spatial']
        for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
            row_ix,col_ix = row_col
        # print(f'{row_ix},{col_ix}')
            g_matrix[row_ix,col_ix] += g[i][0]
        all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
    all_gmat_v_wotm = {}
    label = np.array([])
    for i in range(len(key_v)):
        if key_v[i] not in key_m:
            all_gmat_v_wotm.update({key_v[i]:dataset_v[i]})
            #label = np.append(label, y_pred[i]*0.01)
    key_v_wotm = list(all_gmat_v_wotm.keys())
    dataset_v_wotm = np.array(list(all_gmat_v_wotm.values()))
    print('Shape of the non-overlapping gene set:',dataset_v_wotm.shape)
    
    emb_part = []
    total_samples = len(dataset_v_wotm)
    batch_size = 30
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = dataset_v_wotm[i:i+batch_size]  # Get a batch of data
            batch_data = torch.Tensor(batch_data).unsqueeze(1).to(device)
            _, batch_result_emb, _, _, _ = model(batch_data)
            emb_part.append(batch_result_emb)
    # Concatenate the results along the batch dimension
    emb = torch.cat(emb_part, dim=0)

    print('Shape of SGRs:', emb.shape)
    
    #===========first gen==============
    dataset_vinm_emb=[]
    with torch.no_grad():
        for key in key_m:
            data = all_gmat_v[key]
            data = torch.Tensor(data).unsqueeze(0).to(device)
            data = data.unsqueeze(0)
            _, batch_result_emb, _, _, _ = model(data)
            dataset_vinm_emb.append(batch_result_emb)
        dataset_vinm_emb = torch.cat(dataset_vinm_emb, dim=0)
    dataset_vinm_emb = dataset_vinm_emb.detach().cpu().numpy()
    emb = emb.detach().cpu().numpy()
    sample1 = np.concatenate((dataset_vinm_emb, emb), axis=0)
    correlations = np.corrcoef(sample1)
    coor = np.random.randn(len(dataset_vinm_emb), len(emb))
    
    for i in range(len(dataset_vinm_emb)):
        for j in range(len(emb)):
            coor[i, j] = correlations[i, j+len(dataset_vinm_emb)]
            
    indices = np.argpartition(coor.flatten(), -110)[-110:]
    row_indices, col_indices = np.unravel_index(indices, coor.shape)
    
    add_visium1 = dataset_v_wotm[col_indices]
    add_visium1_emb = emb[col_indices]
    
    img_gen_part1 = []
    total_samples = len(add_visium1_emb)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = add_visium1_emb[i:i+batch_size]  # Get a batch of data
            batch_data = torch.Tensor(batch_data).to(device)
            batch_result_img_gen, _ = ETC_GAN(batch_data)
            batch_result_img_gen = batch_result_img_gen.to('cpu')
            img_gen_part1.append(batch_result_img_gen)
    # Concatenate the results along the batch dimension
    img_gen1 = torch.cat(img_gen_part1, dim=0)
    
    img_gen1 = img_gen1.detach().cpu().numpy()
    img_gen1 = img_gen1.reshape(-1, 281, 204)
    for i in range(len(img_gen1)):
        img_gen1[i][np.where(img_gen1[i]<0)] = 0
        img_gen1[i][np.where(dataset_m[row_indices[i]]==0)] = 0
    
    print('The first imputation finished, get 110 new genes')
    
    #==========second gen===========
    for i in range(len(col_indices)):
        key_m.append(key_v_wotm[col_indices[i]])

    dataset_vinm_emb=[]
    with torch.no_grad():
        for key in key_m:
            data = all_gmat_v[key]
            data = torch.Tensor(data).unsqueeze(0).to(device)
            data = data.unsqueeze(0)
            _, batch_result_emb, _, _, _ = model(data)
            dataset_vinm_emb.append(batch_result_emb)
        dataset_vinm_emb = torch.cat(dataset_vinm_emb, dim=0)
    dataset_vinm_emb = dataset_vinm_emb.detach().cpu().numpy()
    
    mask = np.ones_like(np.random.random(len(emb)), dtype=bool)
    mask[col_indices] = False
    emb = emb[mask]
    
    #emb = emb.detach().cpu().numpy()
    sample2 = np.concatenate((dataset_vinm_emb, emb), axis=0)
    
    correlations = np.corrcoef(sample2)
    
    coor = np.random.randn(len(dataset_vinm_emb), len(emb))
    for i in range(len(dataset_vinm_emb)):
        for j in range(len(emb)):
            coor[i, j] = correlations[i, j+len(dataset_vinm_emb)]
            
    indices = np.argpartition(coor.flatten(), -110)[-110:]
    row_indices, col_indices = np.unravel_index(indices, coor.shape)
    
    add_visium2_emb = emb[col_indices]
    
    img_gen_part2 = []
    total_samples = len(add_visium2_emb)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = add_visium2_emb[i:i+batch_size]  # Get a batch of data
            batch_data = torch.Tensor(batch_data).to(device)
            batch_result_img_gen, _ = ETC_GAN(batch_data)
            batch_result_img_gen = batch_result_img_gen.to('cpu')
            img_gen_part2.append(batch_result_img_gen)
    # Concatenate the results along the batch dimension
    img_gen2 = torch.cat(img_gen_part2, dim=0)
    dataset_m1 = np.concatenate((dataset_m, img_gen1), axis=0)
    
    img_gen2 = img_gen2.detach().cpu().numpy()
    img_gen2 = img_gen2.reshape(-1, 281, 204)
    for i in range(len(img_gen2)):
        img_gen2[i][np.where(img_gen2[i]<0)] = 0
        img_gen2[i][np.where(dataset_m1[row_indices[i]]==0)] = 0
    
    print('The second imputation finished, get 220 new genes')
    #=======third gen========
    for i in range(len(col_indices)):
        key_m.append(key_v_wotm[col_indices[i]])

    dataset_vinm_emb=[]
    with torch.no_grad():
        for key in key_m:
            data = all_gmat_v[key]
            data = torch.Tensor(data).unsqueeze(0).to(device)
            data = data.unsqueeze(0)
            _, batch_result_emb, _, _, _ = model(data)
            dataset_vinm_emb.append(batch_result_emb)
        dataset_vinm_emb = torch.cat(dataset_vinm_emb, dim=0)
    dataset_vinm_emb = dataset_vinm_emb.detach().cpu().numpy()

    mask = np.ones_like(np.random.random(len(emb)), dtype=bool)
    mask[col_indices] = False
    emb = emb[mask]
    sample3 = np.concatenate((dataset_vinm_emb, emb), axis=0)
    
    correlations = np.corrcoef(sample3)
    coor = np.random.randn(len(dataset_vinm_emb), len(emb))
    for i in range(len(dataset_vinm_emb)):
        for j in range(len(emb)):
            coor[i, j] = correlations[i, j+len(dataset_vinm_emb)]
            
    indices = np.argpartition(coor.flatten(), -110)[-110:]
    row_indices, col_indices = np.unravel_index(indices, coor.shape)
    add_visium3_emb = emb[col_indices]
    
    img_gen_part3 = []
    total_samples = len(add_visium3_emb)
    batch_size = 32
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = add_visium3_emb[i:i+batch_size]  # Get a batch of data
            batch_data = torch.Tensor(batch_data).to(device)
            batch_result_img_gen, _ = ETC_GAN(batch_data)
            batch_result_img_gen = batch_result_img_gen.to('cpu')
            img_gen_part3.append(batch_result_img_gen)
    # Concatenate the results along the batch dimension
    img_gen3 = torch.cat(img_gen_part3, dim=0)
    
    dataset_m2 = np.concatenate((dataset_m1, img_gen2), axis=0)
    img_gen3 = img_gen3.detach().cpu().numpy()
    img_gen3 = img_gen3.reshape(-1, 281, 204)
    for i in range(len(img_gen3)):
        img_gen3[i][np.where(img_gen3[i]<0)] = 0
        img_gen3[i][np.where(dataset_m2[row_indices[i]]==0)] = 0
    
    dataset_m3 = np.concatenate((dataset_m2, img_gen3), axis=0)
    for i in range(len(col_indices)):
        key_m.append(key_v_wotm[col_indices[i]])
    
    print('The third imputation finished, get 330 new genes')
    
    return dataset_m3, key_m

def ari_evalution(adata):
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20, use_rep='X')
    sc.tl.umap(adata)
    resolution = find_resolution(adata, 0.1, 1.0, 22)
    sc.tl.leiden(adata, resolution=0.2125)
    # 计算原始nmi
    leiden_seqfish_part = list(adata.obs['leiden'])
    leiden_seqfish_part = np.array(leiden_seqfish_part).astype(int)

    # 字符类别列表
    ground_truth = list(adata.obs['celltype_mapped_refined'])

    # 创建一个映射字典，将字符类别映射到整数类别
    category_to_int = {category: idx for idx, category in enumerate(set(ground_truth))}
    from collections import OrderedDict
    category_to_int = OrderedDict(sorted(category_to_int.items()))
    i = 0
    for key in category_to_int:
        # 访问和修改值
        category_to_int[key] = i
        i = i+1

    # 使用列表推导式将字符类别列表转换为整数类别列表
    ground = [category_to_int[category] for category in ground_truth]
    ari = adjusted_rand_score(ground, leiden_seqfish_part)
    print('ARI:',ari)
    y_true = np.array(ground)
    y_pred = leiden_seqfish_part
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = (np.array(linear_assignment(w.max() - w))).transpose()
    adata.uns['leiden_colors'] = adata.uns['celltype_mapped_refined_colors'][ind[:, 1]]
    sq.pl.spatial_scatter(adata, color="leiden", shape=None, figsize=(5, 10), dpi=500, frameon=False, legend_loc=None, save = 'orign_leiden.png', title=None)