import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from itertools import cycle

def generate_paired_circle_AD(
    rna_save_path,
    adt_save_path,
    # 基础参数
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_adts=30,
    seed=42,
    # RNA噪声参数
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    # ADT噪声参数
    adt_batch_sigma=1.4,
    adt_gamma_shape=3,
    adt_gamma_scale=3.0,
    adt_cross_talk_rate=0.7,
    adt_sample_batches=3,                # 新增：样本批次数
    adt_noise_ratio=(0.7, 0.3, 0.2),     # 新增：信号/背景/泊松混合比例
    adt_overlap_rate=0.5,                # 新增：ADT特征共享概率
    adt_nonlinear_coefs=(1.0, 0.3, 1.5),# 新增：非线性变换系数
    adt_batch_effect_ratio=0.6,          # 新增：样本批次效应强度比例
):
    """生成配对的单细胞RNA和ADT数据，包含全参数化的噪声模型"""
    np.random.seed(seed)
    
    # ========== 生成原始数据 ==========
    # 坐标系统
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])
    center_coords = raw_coords - 0.5
    radii = np.linalg.norm(center_coords, axis=1)

    # 细胞类型分箱
    max_radius = radii.max()
    radius_bins = np.linspace(0, max_radius, n_cell_types + 1)
    radius_bins[-1] += 1e-6
    labels = np.digitize(radii, radius_bins, right=False) - 1
    labels = np.clip(labels, 0, n_cell_types - 1)
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    labels = np.array([cell_types[i] for i in labels])

    # 生成RNA矩阵
    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size**2, n_genes))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(10):
            gene_idx = i * 10 + j
            if gene_idx >= n_genes:
                continue
            spatial_effect = np.exp(-5 * (radii[mask] - radius_bins[i])**2)
            base_expr = np.random.poisson(15 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base_expr + np.random.poisson(2, sum(mask))

    # 生成ADT矩阵
    adt_matrix = np.zeros((grid_size**2, n_adts))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(3):
            adt_idx = i * 3 + j
            if adt_idx >= n_adts:
                continue
            gene_idx = i * 10 + j * 3
            if gene_idx >= n_genes:
                gene_idx = i * 10
            rna_base = rna_matrix[mask, gene_idx] / 8
            adt_matrix[mask, adt_idx] = np.random.lognormal(np.log(rna_base + 1), 0.3)

    # ========== RNA噪声处理 ==========
    rna = rna_matrix.copy()
    # Dropout噪声
    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)
    # 库大小差异
    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

    # ========== 改进的ADT噪声处理 ==========
    adt = adt_matrix.copy()
    labels_arr = labels

    # 1. 双重批次效应
    sample_labels = np.random.choice(adt_sample_batches, size=adt.shape[0])
    
    # 细胞类型级批次
    celltype_batch_effect = np.random.lognormal(0, adt_batch_sigma, (n_cell_types, adt.shape[1]))
    
    # 样本级批次
    sample_batch_effect = np.random.lognormal(
        0, 
        adt_batch_sigma * adt_batch_effect_ratio,
        (adt_sample_batches, adt.shape[1])
    )
    
    for i in range(n_cell_types):
        mask = labels_arr == f"CellType{i}"
        adt[mask] *= celltype_batch_effect[i]
        for s in range(adt_sample_batches):
            sample_mask = mask & (sample_labels == s)
            adt[sample_mask] *= sample_batch_effect[s]

    # 2. 混合噪声模型
    signal_ratio, bg_ratio, poisson_ratio = adt_noise_ratio
    total = signal_ratio + bg_ratio + poisson_ratio
    signal_ratio /= total
    bg_ratio /= total
    poisson_ratio /= total
    
    background = np.random.gamma(adt_gamma_shape, adt_gamma_scale, adt.shape)
    poisson_noise = np.random.poisson(2.0, adt.shape)
    adt = (signal_ratio * adt + 
          bg_ratio * background + 
          poisson_ratio * poisson_noise)

    # 3. 动态抗体串扰
    cross_talk = np.eye(adt.shape[1])
    for i in range(adt.shape[1]):
        targets = np.random.choice(
            adt.shape[1],
            int(adt_cross_talk_rate * adt.shape[1]),
            replace=False
        )
        for j in targets:
            if i != j:
                cross_talk[i,j] = np.random.uniform(0.2, 0.8)
    adt = adt.dot(cross_talk)

    # 4. ADT特征重叠
    for adt_idx in range(n_adts):
        if np.random.rand() < adt_overlap_rate:
            shared_types = np.random.choice(
                n_cell_types,
                np.random.randint(2,4),
                replace=False
            )
            for ct in shared_types:
                mask = labels_arr == f"CellType{ct}"
                adt[mask, adt_idx] += np.abs(np.random.normal(0.5, 0.8, np.sum(mask)))

    # 5. 非线性变换
    sqrt_coef, power_scale, power_exp = adt_nonlinear_coefs
    adt = sqrt_coef * np.sqrt(adt) + power_scale * np.power(adt, power_exp)

    # ========== 数据保存 ==========
    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "adt": pd.DataFrame(adt, columns=[f"ADT_{i}" for i in range(n_adts)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }


    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    adt_adata = ad.AnnData(X=data["adt"].values)
    adt_adata.obsm["spatial"] = data["coordinates"].values
    adt_adata.obs["cell_type"] = data["cell_types"].values


    rna_adata.write_h5ad(rna_save_path)
    adt_adata.write_h5ad(adt_save_path)

    calculate_ari({
        "rna": data['rna'],
        "adt": data['adt'],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })



def generate_paired_bin_AD(
    rna_save_path,
    adt_save_path,
    # 基础参数
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_adts=30,
    seed=42,
    # RNA噪声参数
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    # ADT噪声参数
    adt_batch_sigma=1.4,
    adt_gamma_shape=3,
    adt_gamma_scale=3.0,
    adt_cross_talk_rate=0.7,
    adt_sample_batches=3,               
    adt_noise_ratio=(0.7, 0.3, 0.2),     
    adt_overlap_rate=0.5,               
    adt_nonlinear_coefs=(1.0, 0.3, 1.5),
    adt_batch_effect_ratio=0.6,          
):
    """生成配对的单细胞RNA和ADT数据，添加可控噪声，并保存为H5AD文件"""
    np.random.seed(seed)

    # 生成坐标系统
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 计算斜线投影值（x+y方向）
    projection = raw_coords[:, 0] + raw_coords[:, 1]  # 45度方向投影
    
    # 斜线分箱设置
    max_projection = projection.max()  # 最大投影值（约为2.0）
    min_projection = projection.min()  # 最小投影值（约为0.0）
    projection_bins = np.linspace(min_projection, max_projection, n_cell_types+1)
    projection_bins[-1] += 1e-6  # 确保包含边界
    
    labels = np.digitize(projection, projection_bins, right=False) - 1
    labels = np.clip(labels, 0, n_cell_types-1)
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    labels = np.array([cell_types[i] for i in labels])

    # 生成RNA矩阵
    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size**2, n_genes))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(10):  # 每个类型10个marker基因
            gene_idx = i*10 + j
            # 基于投影值的空间效应（中心强，边缘弱）
            bin_center = (projection_bins[i] + projection_bins[i+1])/2
            spatial_effect = np.exp(-50*(projection[mask] - bin_center)**2)  # 增加系数使条带更锐利
            base_expr = np.random.poisson(20 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base_expr + np.random.poisson(2, sum(mask))
    
    # 生成ADT矩阵（保持原有逻辑）
    adt_matrix = np.zeros((grid_size**2, n_adts))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(3):  # 每个类型3个marker蛋白
            adt_idx = i*3 + j
            rna_base = rna_matrix[mask, i*10 + j*3] / 8
            adt_matrix[mask, adt_idx] = np.random.lognormal(np.log(rna_base + 1), 0.3)
    # ========== 添加噪声 ==========
    # RNA噪声
    rna = rna_matrix.copy()
    adt = adt_matrix.copy()
    labels_arr = labels

    # 1. Dropout噪声
    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)

    # 2. 库大小差异
    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

    # ========== 改进的ADT噪声处理 ==========
    adt = adt_matrix.copy()
    labels_arr = labels

    # 1. 双重批次效应
    sample_labels = np.random.choice(adt_sample_batches, size=adt.shape[0])
    
    # 细胞类型级批次
    celltype_batch_effect = np.random.lognormal(0, adt_batch_sigma, (n_cell_types, adt.shape[1]))
    
    # 样本级批次
    sample_batch_effect = np.random.lognormal(
        0, 
        adt_batch_sigma * adt_batch_effect_ratio,
        (adt_sample_batches, adt.shape[1])
    )
    
    for i in range(n_cell_types):
        mask = labels_arr == f"CellType{i}"
        adt[mask] *= celltype_batch_effect[i]
        for s in range(adt_sample_batches):
            sample_mask = mask & (sample_labels == s)
            adt[sample_mask] *= sample_batch_effect[s]

    # 2. 混合噪声模型
    signal_ratio, bg_ratio, poisson_ratio = adt_noise_ratio
    total = signal_ratio + bg_ratio + poisson_ratio
    signal_ratio /= total
    bg_ratio /= total
    poisson_ratio /= total
    
    background = np.random.gamma(adt_gamma_shape, adt_gamma_scale, adt.shape)
    poisson_noise = np.random.poisson(2.0, adt.shape)
    adt = (signal_ratio * adt + 
          bg_ratio * background + 
          poisson_ratio * poisson_noise)

    # 3. 动态抗体串扰
    cross_talk = np.eye(adt.shape[1])
    for i in range(adt.shape[1]):
        targets = np.random.choice(
            adt.shape[1],
            int(adt_cross_talk_rate * adt.shape[1]),
            replace=False
        )
        for j in targets:
            if i != j:
                cross_talk[i,j] = np.random.uniform(0.2, 0.8)
    adt = adt.dot(cross_talk)

    # 4. ADT特征重叠
    for adt_idx in range(n_adts):
        if np.random.rand() < adt_overlap_rate:
            shared_types = np.random.choice(
                n_cell_types,
                np.random.randint(2,4),
                replace=False
            )
            for ct in shared_types:
                mask = labels_arr == f"CellType{ct}"
                adt[mask, adt_idx] += np.abs(np.random.normal(0.5, 0.8, np.sum(mask)))

    # 5. 非线性变换
    sqrt_coef, power_scale, power_exp = adt_nonlinear_coefs
    adt = sqrt_coef * np.sqrt(adt) + power_scale * np.power(adt, power_exp)


    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "adt": pd.DataFrame(adt, columns=[f"ADT_{i}" for i in range(n_adts)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }


    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    adt_adata = ad.AnnData(X=data["adt"].values)
    adt_adata.obsm["spatial"] = data["coordinates"].values
    adt_adata.obs["cell_type"] = data["cell_types"].values


    rna_adata.write_h5ad(rna_save_path)
    adt_adata.write_h5ad(adt_save_path)

    calculate_ari({
        "rna": data['rna'],
        "adt": data['adt'],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })



def generate_paired_patch_AD(
    rna_save_path,
    adt_save_path,
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_adts=30,
    ellipse_params=None,
    seed=42,
    # RNA噪声参数
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    # ADT噪声参数
    adt_batch_sigma=1.4,
    adt_gamma_shape=3,
    adt_gamma_scale=3.0,
    adt_cross_talk_rate=0.7,
    adt_sample_batches=3,               
    adt_noise_ratio=(0.7, 0.3, 0.2),     
    adt_overlap_rate=0.5,               
    adt_nonlinear_coefs=(1.0, 0.3, 1.5),
    adt_batch_effect_ratio=0.6,  
):
    np.random.seed(seed)
    
    # ========== 坐标系统生成 ==========
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # ========== 椭圆参数处理 ==========
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    background_type = cell_types[-1]
    
    # 自动生成默认椭圆布局
    if ellipse_params is None:
        ellipse_params = _generate_default_ellipses(n_cell_types)
    
    # 参数验证
    _validate_parameters(n_cell_types, ellipse_params)
    
    # ========== 细胞类型分配 ==========
    labels = np.full(raw_coords.shape[0], background_type)
    for i, param in enumerate(ellipse_params[:n_cell_types-1]):
        cx, cy = param['center']
        a, b = param['a'], param['b']
        
        # 计算椭圆内的点（使用椭圆方程）
        distances = ((raw_coords[:, 0] - cx)/a)**2 + ((raw_coords[:, 1] - cy)/b)**2
        mask = (distances <= 1) & (labels == background_type)
        labels[mask] = cell_types[i]

    # ========== RNA矩阵生成 ==========
    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size**2, n_genes))
    
    # 为每个椭圆类型添加空间特异性表达
    for i in range(n_cell_types-1):  # 跳过背景类型
        mask = labels == cell_types[i]
        param = ellipse_params[i]
        cx, cy = param['center']
        a, b = param['a'], param['b']
        
        # 计算归一化椭圆距离
        x_norm = (raw_coords[mask, 0] - cx) / a
        y_norm = (raw_coords[mask, 1] - cy) / b
        distances = np.sqrt(x_norm**2 + y_norm**2)
        
        # 空间效应：高斯衰减
        spatial_effect = np.exp(-5 * distances**2)
        
        # 生成每个类型的marker基因
        for j in range(10):
            gene_idx = i * 10 + j
            if gene_idx >= n_genes:
                break  # 防止基因索引越界
            
            base = np.random.poisson(15 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base + np.random.poisson(2, mask.sum())

    # ========== ADT矩阵生成 ==========
    adt_matrix = np.zeros((grid_size**2, n_adts))
    for i in range(n_cell_types-1):  # 跳过背景类型
        mask = labels == cell_types[i]
        for j in range(3):  # 每个类型3个ADT
            adt_idx = i * 3 + j
            if adt_idx >= n_adts:
                break
            
            # 关联对应的基因表达
            gene_idx = i * 10 + j * 3
            if gene_idx >= n_genes:
                gene_idx = n_genes - 1  # 保护措施
            
            rna_base = rna_matrix[mask, gene_idx] / 8
            adt_matrix[mask, adt_idx] = np.random.lognormal(np.log(rna_base + 1), 0.3)



    # ========== 添加噪声 ==========
    # RNA噪声
    rna = rna_matrix.copy()
    adt = adt_matrix.copy()
    labels_arr = labels

    # 1. Dropout噪声
    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)

    # 2. 库大小差异
    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

     # ========== 改进的ADT噪声处理 ==========
    adt = adt_matrix.copy()
    labels_arr = labels

    # 1. 双重批次效应
    sample_labels = np.random.choice(adt_sample_batches, size=adt.shape[0])
    
    # 细胞类型级批次
    celltype_batch_effect = np.random.lognormal(0, adt_batch_sigma, (n_cell_types, adt.shape[1]))
    
    # 样本级批次
    sample_batch_effect = np.random.lognormal(
        0, 
        adt_batch_sigma * adt_batch_effect_ratio,
        (adt_sample_batches, adt.shape[1])
    )
    
    for i in range(n_cell_types):
        mask = labels_arr == f"CellType{i}"
        adt[mask] *= celltype_batch_effect[i]
        for s in range(adt_sample_batches):
            sample_mask = mask & (sample_labels == s)
            adt[sample_mask] *= sample_batch_effect[s]

    # 2. 混合噪声模型
    signal_ratio, bg_ratio, poisson_ratio = adt_noise_ratio
    total = signal_ratio + bg_ratio + poisson_ratio
    signal_ratio /= total
    bg_ratio /= total
    poisson_ratio /= total
    
    background = np.random.gamma(adt_gamma_shape, adt_gamma_scale, adt.shape)
    poisson_noise = np.random.poisson(2.0, adt.shape)
    adt = (signal_ratio * adt + 
          bg_ratio * background + 
          poisson_ratio * poisson_noise)

    # 3. 动态抗体串扰
    cross_talk = np.eye(adt.shape[1])
    for i in range(adt.shape[1]):
        targets = np.random.choice(
            adt.shape[1],
            int(adt_cross_talk_rate * adt.shape[1]),
            replace=False
        )
        for j in targets:
            if i != j:
                cross_talk[i,j] = np.random.uniform(0.2, 0.8)
    adt = adt.dot(cross_talk)

    # 4. ADT特征重叠
    for adt_idx in range(n_adts):
        if np.random.rand() < adt_overlap_rate:
            shared_types = np.random.choice(
                n_cell_types,
                np.random.randint(2,4),
                replace=False
            )
            for ct in shared_types:
                mask = labels_arr == f"CellType{ct}"
                adt[mask, adt_idx] += np.abs(np.random.normal(0.5, 0.8, np.sum(mask)))

    # 5. 非线性变换
    sqrt_coef, power_scale, power_exp = adt_nonlinear_coefs
    adt = sqrt_coef * np.sqrt(adt) + power_scale * np.power(adt, power_exp)


    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "adt": pd.DataFrame(adt, columns=[f"ADT_{i}" for i in range(n_adts)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }


    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    adt_adata = ad.AnnData(X=data["adt"].values)
    adt_adata.obsm["spatial"] = data["coordinates"].values
    adt_adata.obs["cell_type"] = data["cell_types"].values


    rna_adata.write_h5ad(rna_save_path)
    adt_adata.write_h5ad(adt_save_path)

    calculate_ari({
        "rna": data['rna'],
        "adt": data['adt'],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })

def _generate_default_ellipses(n_cell_types):
    """生成默认椭圆布局（花瓣状分布）"""
    n_ellipses = n_cell_types - 1
    if n_ellipses <= 0:
        return []
    
    # 自动分配位置（圆形布局）
    angles = np.linspace(0, 2*np.pi, n_ellipses, endpoint=False)
    return [{
        'center': (0.5 + 0.3*np.cos(a), 0.5 + 0.3*np.sin(a)),
        'a': 0.12 + 0.03*i,
        'b': 0.12 + 0.03*(i%2)
    } for i, a in enumerate(angles)]

def _validate_parameters(n_cell_types, ellipse_params):
    """参数验证"""
    required_ellipses = n_cell_types - 1
    if len(ellipse_params) < required_ellipses:
        raise ValueError(
            f"需要至少{required_ellipses}个椭圆参数，当前提供{len(ellipse_params)}个"
        )
    
    for i, param in enumerate(ellipse_params[:required_ellipses]):
        if not {'center', 'a', 'b'}.issubset(param.keys()):
            raise KeyError(f"椭圆参数#{i}必须包含center, a, b三个键")
        
        cx, cy = param['center']
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            raise ValueError(f"椭圆#{i}中心坐标必须在[0,1]范围内")
        
        a, b = param['a'], param['b']
        if a <= 0 or b <= 0:
            raise ValueError(f"椭圆#{i}轴长必须大于0")


def generate_paired_circle_ATAC(
    rna_save_path,
    atac_save_path,
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_atacs=30,
    seed=42,
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    atac_noise_flip_prob=0.1,  # 控制 0/1 翻转的概率
    atac_overlap_rate=0.5
):
    """生成配对的单细胞RNA和ATAC数据，包含二值ATAC噪声模型"""
    np.random.seed(seed)

    # ======= 空间坐标和细胞类型生成 =======
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])
    center_coords = raw_coords - 0.5
    radii = np.linalg.norm(center_coords, axis=1)

    max_radius = radii.max()
    radius_bins = np.linspace(0, max_radius, n_cell_types + 1)
    radius_bins[-1] += 1e-6
    labels = np.digitize(radii, radius_bins, right=False) - 1
    labels = np.clip(labels, 0, n_cell_types - 1)
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    labels = np.array([cell_types[i] for i in labels])

    # ======= RNA生成 =======
    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size**2, n_genes))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(10):
            gene_idx = i * 10 + j
            if gene_idx >= n_genes:
                continue
            spatial_effect = np.exp(-5 * (radii[mask] - radius_bins[i])**2)
            base_expr = np.random.poisson(15 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base_expr + np.random.poisson(2, sum(mask))

    # ======= RNA噪声 =======
    rna = rna_matrix.copy()
    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)
    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

    # ======= ATAC生成（0/1） =======
    atac_matrix = np.zeros((grid_size**2, n_atacs), dtype=np.int32)
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(3):
            atac_idx = i * 3 + j
            if atac_idx >= n_atacs:
                continue
            gene_idx = i * 10 + j * 3
            if gene_idx >= n_genes:
                gene_idx = i * 10
            # 根据RNA表达水平决定ATAC开启概率
            rna_base = rna_matrix[mask, gene_idx]
            prob = np.clip(rna_base / (rna_base.max() + 1e-6), 0, 1)
            atac_matrix[mask, atac_idx] = np.random.rand(np.sum(mask)) < prob

    # ======= ATAC噪声处理 =======
    atac = atac_matrix.copy()
    # 1. 随机翻转部分位点（simulated technical noise）
    flip_mask = np.random.rand(*atac.shape) < atac_noise_flip_prob
    atac = np.logical_xor(atac, flip_mask).astype(np.int32)

    # 2. ATAC特征重叠（不同细胞类型共享某些peak）
    for atac_idx in range(n_atacs):
        if np.random.rand() < atac_overlap_rate:
            shared_types = np.random.choice(
                n_cell_types,
                np.random.randint(2, 4),
                replace=False
            )
            for ct in shared_types:
                mask = labels == f"CellType{ct}"
                # 对已有值中随机一部分设置为1
                n_mask = np.sum(mask)
                prob_share = 0.2 + 0.5 * np.random.rand()
                shared_bits = np.random.rand(n_mask) < prob_share
                atac[mask, atac_idx] = np.maximum(atac[mask, atac_idx], shared_bits.astype(np.int32))

    # ======= 数据保存 =======
    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "atac": pd.DataFrame(atac, columns=[f"ATAC_{i}" for i in range(n_atacs)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }

    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    atac_adata = ad.AnnData(X=data["atac"].values)
    atac_adata.obsm["spatial"] = data["coordinates"].values
    atac_adata.obs["cell_type"] = data["cell_types"].values

    rna_adata.write_h5ad(rna_save_path)
    atac_adata.write_h5ad(atac_save_path)

    calculate_ari({
        "rna": data["rna"],
        "adt": data["atac"],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })

def generate_paired_bin_ATAC(
    rna_save_path,
    atac_save_path,
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_atacs=30,
    seed=42,
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    atac_batch_sigma=1.4,
    atac_gamma_shape=3,
    atac_gamma_scale=3.0,
    atac_cross_talk_rate=0.7,
    atac_sample_batches=3,               
    atac_noise_ratio=(0.7, 0.3, 0.2),     
    atac_overlap_rate=0.5,               
    atac_nonlinear_coefs=(1.0, 0.3, 1.5),
    atac_batch_effect_ratio=0.6,  
):
    np.random.seed(seed)

    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    projection = raw_coords[:, 0] + raw_coords[:, 1]
    max_projection = projection.max()
    min_projection = projection.min()
    projection_bins = np.linspace(min_projection, max_projection, n_cell_types + 1)
    projection_bins[-1] += 1e-6
    
    labels = np.digitize(projection, projection_bins, right=False) - 1
    labels = np.clip(labels, 0, n_cell_types - 1)
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    labels = np.array([cell_types[i] for i in labels])

    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size**2, n_genes))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(10):
            gene_idx = i * 10 + j
            bin_center = (projection_bins[i] + projection_bins[i + 1]) / 2
            spatial_effect = np.exp(-50 * (projection[mask] - bin_center)**2)
            base_expr = np.random.poisson(20 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base_expr + np.random.poisson(2, sum(mask))

    atac_matrix = np.zeros((grid_size**2, n_atacs))
    for i in range(n_cell_types):
        mask = labels == cell_types[i]
        for j in range(3):
            atac_idx = i * 3 + j
            rna_base = rna_matrix[mask, i * 10 + j * 3] / 8
            atac_matrix[mask, atac_idx] = np.random.lognormal(np.log(rna_base + 1), 0.3)

    rna = rna_matrix.copy()
    atac = atac_matrix.copy()

    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)

    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

    sample_labels = np.random.choice(atac_sample_batches, size=atac.shape[0])
    celltype_batch_effect = np.random.lognormal(0, atac_batch_sigma, (n_cell_types, atac.shape[1]))
    sample_batch_effect = np.random.lognormal(0, atac_batch_sigma * atac_batch_effect_ratio, (atac_sample_batches, atac.shape[1]))
    for i in range(n_cell_types):
        mask = labels == f"CellType{i}"
        atac[mask] *= celltype_batch_effect[i]
        for s in range(atac_sample_batches):
            sample_mask = mask & (sample_labels == s)
            atac[sample_mask] *= sample_batch_effect[s]

    signal_ratio, bg_ratio, poisson_ratio = atac_noise_ratio
    total = signal_ratio + bg_ratio + poisson_ratio
    signal_ratio /= total
    bg_ratio /= total
    poisson_ratio /= total
    
    background = np.random.gamma(atac_gamma_shape, atac_gamma_scale, atac.shape)
    poisson_noise = np.random.poisson(2.0, atac.shape)
    atac = (signal_ratio * atac +
            bg_ratio * background +
            poisson_ratio * poisson_noise)

    cross_talk = np.eye(atac.shape[1])
    for i in range(atac.shape[1]):
        targets = np.random.choice(atac.shape[1], int(atac_cross_talk_rate * atac.shape[1]), replace=False)
        for j in targets:
            if i != j:
                cross_talk[i, j] = np.random.uniform(0.2, 0.8)
    atac = atac.dot(cross_talk)

    for atac_idx in range(n_atacs):
        if np.random.rand() < atac_overlap_rate:
            shared_types = np.random.choice(n_cell_types, np.random.randint(2, 4), replace=False)
            for ct in shared_types:
                mask = labels == f"CellType{ct}"
                atac[mask, atac_idx] += np.abs(np.random.normal(0.5, 0.8, np.sum(mask)))

    sqrt_coef, power_scale, power_exp = atac_nonlinear_coefs
    atac = sqrt_coef * np.sqrt(atac) + power_scale * np.power(atac, power_exp)

    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "atac": pd.DataFrame(atac, columns=[f"ATAC_{i}" for i in range(n_atacs)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }

    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    atac_adata = ad.AnnData(X=data["atac"].values)
    atac_adata.obsm["spatial"] = data["coordinates"].values
    atac_adata.obs["cell_type"] = data["cell_types"].values

    rna_adata.write_h5ad(rna_save_path)
    atac_adata.write_h5ad(atac_save_path)

    calculate_ari({
        "rna": data['rna'],
        "adt": data['atac'],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })


def generate_paired_patch_ATAC(
    rna_save_path,
    atac_save_path,
    grid_size=40,
    n_cell_types=10,
    n_genes=200,
    n_atacs=30,
    ellipse_params=None,
    seed=42,
    # RNA噪声参数
    rna_dropout_rate=0.5,
    rna_library_sigma=0.1,
    # ATAC噪声参数
    atac_batch_sigma=1.4,
    atac_gamma_shape=3,
    atac_gamma_scale=3.0,
    atac_cross_talk_rate=0.7,
    atac_sample_batches=3,               
    atac_noise_ratio=(0.7, 0.3, 0.2),     
    atac_overlap_rate=0.5,               
    atac_nonlinear_coefs=(1.0, 0.3, 1.5),
    atac_batch_effect_ratio=0.6,  
):
    np.random.seed(seed)

    # ========== 坐标系统生成 ==========
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    raw_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # ========== 椭圆参数处理 ==========
    cell_types = [f"CellType{i}" for i in range(n_cell_types)]
    background_type = cell_types[-1]

    if ellipse_params is None:
        ellipse_params = _generate_default_ellipses(n_cell_types)

    _validate_parameters(n_cell_types, ellipse_params)

    # ========== 细胞类型分配 ==========
    labels = np.full(raw_coords.shape[0], background_type)
    for i, param in enumerate(ellipse_params[:n_cell_types - 1]):
        cx, cy = param['center']
        a, b = param['a'], param['b']
        distances = ((raw_coords[:, 0] - cx) / a) ** 2 + ((raw_coords[:, 1] - cy) / b) ** 2
        mask = (distances <= 1) & (labels == background_type)
        labels[mask] = cell_types[i]

    # ========== RNA矩阵生成 ==========
    rna_matrix = np.random.negative_binomial(n=5, p=0.8, size=(grid_size ** 2, n_genes))
    for i in range(n_cell_types - 1):
        mask = labels == cell_types[i]
        param = ellipse_params[i]
        cx, cy = param['center']
        a, b = param['a'], param['b']
        x_norm = (raw_coords[mask, 0] - cx) / a
        y_norm = (raw_coords[mask, 1] - cy) / b
        distances = np.sqrt(x_norm ** 2 + y_norm ** 2)
        spatial_effect = np.exp(-5 * distances ** 2)
        for j in range(10):
            gene_idx = i * 10 + j
            if gene_idx >= n_genes:
                break
            base = np.random.poisson(15 * spatial_effect + 2)
            rna_matrix[mask, gene_idx] = base + np.random.poisson(2, mask.sum())

    # ========== ATAC矩阵生成 ==========
    atac_matrix = np.zeros((grid_size ** 2, n_atacs))
    for i in range(n_cell_types - 1):
        mask = labels == cell_types[i]
        for j in range(3):
            atac_idx = i * 3 + j
            if atac_idx >= n_atacs:
                break
            gene_idx = i * 10 + j * 3
            if gene_idx >= n_genes:
                gene_idx = n_genes - 1
            rna_base = rna_matrix[mask, gene_idx] / 8
            atac_matrix[mask, atac_idx] = np.random.lognormal(np.log(rna_base + 1), 0.3)

    # ========== 添加噪声 ==========
    rna = rna_matrix.copy()
    atac = atac_matrix.copy()
    labels_arr = labels

    dropout_mask = np.random.rand(*rna.shape) < rna_dropout_rate
    rna = rna * (1 - dropout_mask)

    library_factors = np.random.lognormal(0, rna_library_sigma, rna.shape[0])
    rna = rna * library_factors[:, None]

    atac = atac_matrix.copy()
    sample_labels = np.random.choice(atac_sample_batches, size=atac.shape[0])
    celltype_batch_effect = np.random.lognormal(0, atac_batch_sigma, (n_cell_types, atac.shape[1]))
    sample_batch_effect = np.random.lognormal(
        0,
        atac_batch_sigma * atac_batch_effect_ratio,
        (atac_sample_batches, atac.shape[1])
    )
    for i in range(n_cell_types):
        mask = labels_arr == f"CellType{i}"
        atac[mask] *= celltype_batch_effect[i]
        for s in range(atac_sample_batches):
            sample_mask = mask & (sample_labels == s)
            atac[sample_mask] *= sample_batch_effect[s]

    signal_ratio, bg_ratio, poisson_ratio = atac_noise_ratio
    total = signal_ratio + bg_ratio + poisson_ratio
    signal_ratio /= total
    bg_ratio /= total
    poisson_ratio /= total

    background = np.random.gamma(atac_gamma_shape, atac_gamma_scale, atac.shape)
    poisson_noise = np.random.poisson(2.0, atac.shape)
    atac = (signal_ratio * atac +
            bg_ratio * background +
            poisson_ratio * poisson_noise)

    cross_talk = np.eye(atac.shape[1])
    for i in range(atac.shape[1]):
        targets = np.random.choice(
            atac.shape[1],
            int(atac_cross_talk_rate * atac.shape[1]),
            replace=False
        )
        for j in targets:
            if i != j:
                cross_talk[i, j] = np.random.uniform(0.2, 0.8)
    atac = atac.dot(cross_talk)

    for atac_idx in range(n_atacs):
        if np.random.rand() < atac_overlap_rate:
            shared_types = np.random.choice(
                n_cell_types,
                np.random.randint(2, 4),
                replace=False
            )
            for ct in shared_types:
                mask = labels_arr == f"CellType{ct}"
                atac[mask, atac_idx] += np.abs(np.random.normal(0.5, 0.8, np.sum(mask)))

    sqrt_coef, power_scale, power_exp = atac_nonlinear_coefs
    atac = sqrt_coef * np.sqrt(atac) + power_scale * np.power(atac, power_exp)

    data = {
        "rna": pd.DataFrame(rna, columns=[f"Gene_{i}" for i in range(n_genes)]),
        "atac": pd.DataFrame(atac, columns=[f"ATAC_{i}" for i in range(n_atacs)]),
        "coordinates": pd.DataFrame(raw_coords, columns=["x", "y"]),
        "cell_types": pd.Series(labels, name="cell_type")
    }

    rna_adata = ad.AnnData(X=data["rna"].values)
    rna_adata.obsm["spatial"] = data["coordinates"].values
    rna_adata.obs["cell_type"] = data["cell_types"].values

    atac_adata = ad.AnnData(X=data["atac"].values)
    atac_adata.obsm["spatial"] = data["coordinates"].values
    atac_adata.obs["cell_type"] = data["cell_types"].values

    rna_adata.write_h5ad(rna_save_path)
    atac_adata.write_h5ad(atac_save_path)

    calculate_ari({
        "rna": data['rna'],
        "adt": data['atac'],
        "coordinates": data["coordinates"],
        "cell_types": data["cell_types"]
    })


def calculate_ari(noisy_data):
    # RNA处理
    rna_log = np.log1p(noisy_data["rna"])
    rna_pca = PCA(n_components=20).fit_transform(rna_log)
    rna_clusters = KMeans(n_clusters=5, n_init=20).fit_predict(rna_pca)
    rna_ari = adjusted_rand_score(noisy_data["cell_types"], rna_clusters)
    
    # ADT处理
    # ADT预处理改用Robust Scaling
    from sklearn.preprocessing import RobustScaler
    adt_scaled = RobustScaler().fit_transform(noisy_data["adt"])
    adt_pca = PCA(n_components=15).fit_transform(adt_scaled)  # 增加主成分数
    
    # 使用GMM代替KMeans
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=5, n_init=3)
    adt_clusters = gmm.fit_predict(adt_pca)
    adt_ari = adjusted_rand_score(noisy_data["cell_types"], adt_clusters)
    
    print("rna ari:",rna_ari, "adt ari:",adt_ari)
