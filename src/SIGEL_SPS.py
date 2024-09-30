from scipy.stats import nbinom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scanpy as sc
import SpaGCN as spg
import numpy as np
import anndata as ad
import torch
import random
from scipy.sparse import csr_matrix
import statsmodels.api as sm

class preprocess:
    def __init__(self, seed = None):
        self.seed = seed
        pass

    def get_data(self,sample_id):
            assert sample_id == '151676', "please choose the 151676 for testing"
            path_file = f"151676_10xvisium.h5ad"
            adata =  sc.read_h5ad(path_file)
            return adata

    def size_factor_normalization(self,adata):
            adata_X_dense = adata.X.todense()
            adata_X_dense = np.log(adata_X_dense)
            adata_X_dense[~np.isfinite(adata_X_dense)] = 0
            col_means = np.mean(adata_X_dense, axis=0)
            adata_X_dense = adata_X_dense
            size_factor = np.exp(np.median(adata_X_dense, axis=1))
            adata_X_dense = adata.X.todense() / size_factor #The original `adata.X` should be used as the dividend
            adata.X = csr_matrix(adata_X_dense)
            return adata
    
    
    def data_process(self,adata):
            adata = adata[~adata.obs['layer_guess_reordered_short'].isna()]
            adata = adata[~adata.obs['discard'].astype(bool), :]
            adata.obs['cluster']  = adata.obs['layer_guess_reordered_short']
            adata.var_names_make_unique()
            spg.prefilter_genes(adata, min_cells=10)  # avoiding all genes are zeros
            spg.prefilter_specialgenes(adata)
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            adata = adata[:,adata.var['total_counts']>30]
            adata = self.size_factor_normalization(adata)
            adata.obs['array_x']=np.ceil((adata.obs['array_col']-
                                      adata.obs['array_col'].min())/2).astype(int)
            adata.obs['array_y']=adata.obs['array_row']-adata.obs['array_row'].min()
            return adata


class sim_spatial_pattern:
    def __init__(self, seed = None):
        self.seed = seed
        pass

    def estimate_s_cv(self, adata):
        X = adata.X.todense()
        mu = np.mean(X, axis=0)
        mu = np.asarray(mu).flatten()
        sigma_squared = np.var(X, axis=0, ddof=1)
        sigma_squared = np.asarray(sigma_squared).flatten()
        cv_squared = sigma_squared / mu**2
    
        return mu, cv_squared
    
    def fit_glm(self, cv_squared, mu):
        X = 1 / mu
        y = cv_squared
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.identity()))
        results = model.fit()
    
        alpha0 = results.params[0]
        alpha1 = results.params[1]
    
        return alpha0, alpha1

    def simulate(self, adata: ad.AnnData, region_key: str, region_list: list, t: list):
        if self.seed != None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.backends.cudnn.deterministic = True
        
        #======estimate alpha0 and alpha1 for s calculation=====
        sim_data = adata
        mu, cv_squared = self.estimate_s_cv(sim_data)
        alpha0, alpha1 = self.fit_glm(cv_squared, mu)
        for i in range(len(t)):
            
            #========calcalate t-specific target mu=======
            region = region_list[i]
            t_i = t[i]
            section = adata[adata.obs[region_key]==region].X
            dense_matrix = section.todense()
            mu = np.mean(dense_matrix, axis=0)
            mu = np.asarray(mu).flatten()
            sorted_indices = np.argsort(mu)[::-1]
            top_t_percent_count = int(t_i * 0.01 * len(sorted_indices))
            target_mu = mu[sorted_indices[top_t_percent_count]]
            
            #========calcalate t-specific target scv and s=======
            target_cv_squared = alpha0 + alpha1 / target_mu
            s = 1 / (target_cv_squared - 1 / target_mu)
            
            #=======argsrt spot in target region=========
            region_expression = np.median(dense_matrix, axis=1)
            region_expression = np.asarray(region_expression).flatten()
            sorted_spots = np.argsort(region_expression)
            
            
            #============simulate NB distribution===========
            n_spots = len(region_expression)
            p = s / (target_mu + s)
            n = target_mu * p / (1 - p)
            simulated_expression = np.random.negative_binomial(n, p, size=n_spots)
            simulated_expression_sorted = np.sort(simulated_expression)
            
            #===========rearrange simlated spot expression======
            simulated_gene_expression = np.zeros_like(sorted_spots)
            for i, spot in enumerate(sorted_spots):
                simulated_gene_expression[spot] = simulated_expression_sorted[i]
            
            sim_data[sim_data.obs[region_key] == region, sim_data.var_names[0]].X = simulated_gene_expression
            print(f'{region} has been simulated.')
        sim_data.uns = adata.uns
        return sim_data