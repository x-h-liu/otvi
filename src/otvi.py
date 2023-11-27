import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
import scanpy as sc
import scvi
from scipy.spatial import distance
import ot
import torch
import gc


def ot_align(M, D_A, D_B):
    assert M.shape[0] == D_A.shape[0]
    assert M.shape[1] == D_B.shape[0]
    assert D_A.shape[0] == D_A.shape[1] and D_B.shape[0] == D_B.shape[1]
    M = np.copy(M)
    D_A = np.copy(D_A)
    D_B = np.copy(D_B)
    # normalize
    D_A /= D_A[D_A>0].max()
    D_A *= M.max()
    D_B /= D_B[D_B>0].max()
    D_B *= M.max()
    # initial distribution
    p = np.ones((M.shape[0],)) / M.shape[0]
    q = np.ones((M.shape[1],)) / M.shape[1]
    # transport
    pi, logw = ot.gromov.fused_gromov_wasserstein(M, D_A, D_B, p, q, loss_fun='square_loss', alpha=0.1, log=True, verbose=True, numItermax=200, numItermaxEmd=1000000)
    return pi


def integrate(RNA_slice, Epi_slice, center_coordinates, max_iter=30, f_center=16):
    """
    Run the spatial multi-omics integration algorithm.

    Parameters:
    RNA_slice: AnnData object
        The AnnData object of the first slice with an omics measurement, with .obsm['spatial'] field storing the spatial coordinates.
        Note: It doesn't have to be of RNA modality. Any modality is fine.
    Epi_slice: AnnData object
        The AnnData object of the second slice with a different kinds of omics measurement, with .obsm['spatial'] field storing the spatial coordinates.
    center_coordinates: Numpy array of shape (n_center, 2)
        A numpy array storing the x-y coordinates of spots on the center slice.
    max_iter: integer
        Iterations of optimization. Default is 30.
    f_center: integer
        Number of dimensions of cell embedding on the center slice. Default is 16.
    max_iter: int
        The maximal number of iterations DeST-OT is run. Default is 100.

    Returns:
    pi_rna: Numpy array of shape (n_1 x n_center)
        The alignment matrix between spots of the first slice and spots of the center slice.
    pi_epi: Numpy array of shape (n_2 x n_center)
        The alignment matrix between spots of the second slice and spots of the center slice.
    center_embedding: Numpy array of shape (n_center, f_center)
        The cell embedding of each spot on the center slice.
    mvi: MultiVI model
        Trained MultiVI model of the last iteration.
    """
    n_rna = RNA_slice.shape[0]
    n_epi = Epi_slice.shape[0]
    n_center = center_coordinates.shape[0]
    f_rna = RNA_slice.shape[1]
    f_epi = Epi_slice.shape[1]

    # Initialization
    g_rna = np.ones((n_rna,)) / n_rna
    g_epi = np.ones((n_epi,)) / n_epi
    g_center = np.ones((n_center,)) / n_center

    pi_rna = np.outer(g_rna, g_center)
    pi_epi = np.outer(g_epi, g_center)

    # Calculate spatial distances within each slice
    D_rna = distance.cdist(RNA_slice.obsm['spatial'], RNA_slice.obsm['spatial'])
    D_epi = distance.cdist(Epi_slice.obsm['spatial'], Epi_slice.obsm['spatial'])
    D_center = distance.cdist(center_coordinates, center_coordinates)
    
    iter = 0
    while iter < max_iter:
        print(f"=================================ITERATION {iter}=================================")
        # Find embeddings of the center slice
        center_rna = (n_center * pi_rna.T) @ RNA_slice.X
        center_epi = (n_center * pi_epi.T) @ Epi_slice.X
        center_rna_epi_stack = np.hstack((center_rna, center_epi))
        center_adata = ad.AnnData(center_rna_epi_stack, dtype=np.float32)
        center_adata.obsm['spatial'] = center_coordinates
        scvi.model.MULTIVI.setup_anndata(adata=center_adata)
        # train MULTIVI
        mvi = scvi.model.MULTIVI(center_adata, n_genes=f_rna, n_regions=f_epi, n_latent=f_center)
        mvi.train(use_gpu=True)
        center_embedding = mvi.get_latent_representation() # should be of size n_center x 16
        # Find embeddings of the RNA and ATAC slices respectively
        rna_adata_for_latent_extraction = ad.AnnData(np.hstack((np.copy(RNA_slice.X), np.zeros((n_rna, f_epi)))), dtype=np.float32)
        epi_adata_for_latent_extraction = ad.AnnData(np.hstack((np.zeros((n_epi, f_rna)), np.copy(Epi_slice.X))), dtype=np.float32)
        mvi.setup_anndata(adata=rna_adata_for_latent_extraction)
        mvi.setup_anndata(adata=epi_adata_for_latent_extraction)
        rna_embedding = mvi.get_latent_representation(adata=rna_adata_for_latent_extraction, modality='expression') # should be of size n_rna x f_center
        epi_embedding = mvi.get_latent_representation(adata=epi_adata_for_latent_extraction, modality='accessibility') # should be of size n_epi x f_center
        # Calculate feature dissimilarity for both RNA-center and Epi-center
        M_rna = distance.cdist(rna_embedding, center_embedding)
        M_epi = distance.cdist(epi_embedding, center_embedding)
        # update pi
        pi_rna = ot_align(M_rna, D_rna, D_center)
        pi_epi = ot_align(M_epi, D_epi, D_center)
        iter += 1
        # clear cache
        del mvi
        gc.collect()
        torch.cuda.empty_cache()
    # Train final MVI
    center_rna = (n_center * pi_rna.T) @ RNA_slice.X
    center_epi = (n_center * pi_epi.T) @ Epi_slice.X
    center_rna_epi_stack = np.hstack((center_rna, center_epi))
    center_adata = ad.AnnData(center_rna_epi_stack, dtype=np.float32)
    scvi.model.MULTIVI.setup_anndata(adata=center_adata)
    mvi = scvi.model.MULTIVI(center_adata, n_genes=f_rna, n_regions=f_epi, n_latent=f_center)
    mvi.train(use_gpu=True)
    center_embedding = mvi.get_latent_representation() # should be of size n_center x f_center
    return pi_rna, pi_epi, center_embedding, mvi








