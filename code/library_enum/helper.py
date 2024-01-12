import numpy as np
from scipy.spatial.distance import euclidean
from rdkit import Chem, SimDivFilters, DataStructs
from openeye import oechem


def uniform_sampling(df, cutoff, seed):
    # source: https://gis.stackexchange.com/questions/436908/selecting-n-samples-uniformly-from-a-grid-points
    first = True
    list_ok = []
    ind_ok = []

    DISTANCE_MIN = cutoff
    # Shuffle data set based on random seed
    df_mix = df.sample(frac=1, random_state=seed)

    for index, row in df_mix.iterrows():
        if first:
            list_ok.append([row['X'], row['Y']])
            ind_ok.append(index)
            first = False
            continue

        point = [row['X'], row['Y']]
        if any((euclidean(point, point_ok) < DISTANCE_MIN for point_ok in list_ok)):
            continue

        list_ok.append(point)
        ind_ok.append(index)

    return ind_ok

### Iterate until desired number is achieved with a ~10% overshooting rate
## Increasing cutoff decreases number of samples
# This is a barebones MVP, susceptible to oscillating or convergence issues

def uniform_selection(df, N, seed):
    sample = df
    cutoff = 0.5
    i = 0

    while (len(sample) > N*1.1) or (len(sample) < N):
        if len(sample) > N*1.1: # increase cutoff
            cutoff += 0.05
            sample = uniform_sampling(df, cutoff, seed)
        elif len(sample) < N: # decrease cutoff
            cutoff -= 0.05
            sample = uniform_sampling(df, cutoff, seed)
        i += 1
        if i > 5:
            sample_new = sample[:N]
            return df.loc[sample_new]
    sample_new = sample[:N]

    return df.loc[sample_new]

def random_selection(df, N, seed):
    return df.sample(n=N, random_state=seed)

def diversity_selection(df, N, seed):
    mmp = SimDivFilters.MaxMinPicker()
    picks = mmp.LazyBitVectorPick(objects=list(df['fp']), poolSize=len(df), pickSize=N, seed=seed)
    ind = [x for x in picks]
    return df.iloc[ind]

def lib_enum(pamine_selection, cooh_selection):
    libgen = oechem.OELibraryGen('[#6:1][N:2]([H:3])[H:4].[#6:10](=[O:11])[O:12][H:13]>>[#6:1][N:2]([H:3])[#6:10](=[O:11])')

    products = []
    tracker = {}
    for i in range(len(pamine_selection)):
        for j in range(len(cooh_selection)):
            pamine_bb = pamine_selection['BB_SMILES'].iloc[i]
            cooh_bb = cooh_selection['BB_SMILES'].iloc[j]

            mol = oechem.OEGraphMol()
            oechem.OEParseSmiles(mol, pamine_bb)
            libgen.SetStartingMaterial(mol, 0)

            mol.Clear()
            oechem.OEParseSmiles(mol, cooh_bb)
            libgen.SetStartingMaterial(mol, 1)

            k = 0
            for index, product in enumerate(libgen.GetProducts()):
                if index == 0:
                    smi = oechem.OECreateCanSmiString(product)
                    products.append(smi)
                    k += 1

            tracker[f'({i},{j})'] = k
    return products, tracker

def sim_mat(ref_df, test_df):
    sim_matrix = np.zeros((len(ref_df), len(test_df)), dtype=np.float32)
    for i, ref in enumerate(list(ref_df['fp'])):
        for j, test in enumerate(list(test_df['fp'])):
            sim_matrix[i,j] = DataStructs.TanimotoSimilarity(ref,test)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def plot_library_stats(pamine_sele_1, cooh_sele_1, lib1, lib1_label=None,
                 pamine_sele_2=None, cooh_sele_2=None, lib2=None, lib2_label=None,
                 pamine_sele_3=None, cooh_sele_3=None, lib3=None, lib3_label=None):

    library_1 = pd.DataFrame(lib1, columns=["SMILES"])

    # Calculate PCP of enumerated products
    library_1 = calc_pcp(library_1)

    # Calculate cost of BBs
    l1p1_cost = np.sum(pamine_sele_1['Price_250mg'])
    l1p2_cost = np.sum(cooh_sele_1['Price_250mg'])

    # Calculate similarity of BBs
    l1p1_sim = sim_mat(pamine_sele_1, pamine_sele_1)
    l1p2_sim = sim_mat(cooh_sele_1, cooh_sele_1)

    if lib2:
        library_2 = pd.DataFrame(lib2, columns=["SMILES"])
        library_2 = calc_pcp(library_2)

        # Calculate PCP of enumerated products
        library_2 = calc_pcp(library_2)

        # Calculate cost of BBs
        l2p1_cost = np.sum(pamine_sele_2['Price_250mg'])
        l2p2_cost = np.sum(cooh_sele_2['Price_250mg'])

        # Calculate similarity of BBs
        l2p1_sim = sim_mat(pamine_sele_2, pamine_sele_2)
        l2p2_sim = sim_mat(cooh_sele_2, cooh_sele_2)


    if lib3:
        library_3 = pd.DataFrame(lib3, columns=["SMILES"])
        library_3 = calc_pcp(library_3)

        # Calculate PCP of enumerated products
        library_3 = calc_pcp(library_3)

        # Calculate cost of BBs
        l3p1_cost = np.sum(pamine_sele_3['Price_250mg'])
        l3p2_cost = np.sum(cooh_sele_3['Price_250mg'])

        # Calculate similarity of BBs
        l3p1_sim = sim_mat(pamine_sele_3, pamine_sele_3)
        l3p2_sim = sim_mat(cooh_sele_3, cooh_sele_3)


    fig, axs = plt.subplots(4, 2, figsize=(10,24), dpi=150)
    fig.subplots_adjust(wspace=0.3)

    axs[0][0].hist(library_1['mw'], bins=20, alpha=0.5, density=True)
    axs[0][0].axvline(x=500, color='black', linestyle='dashed')

    if lib2:
        axs[0][0].hist(library_2['mw'], bins=20, alpha=0.5, density=True, color='orange')

    if lib3:
        axs[0][0].hist(library_3['mw'], bins=20, alpha=0.5, density=True, color='gray')

    axs[0][0].set_xlabel('MW (Da)')
    axs[0][0].set_xlim([100, 600])

    axs[0][1].hist(library_1['xlogp'], bins=20, alpha=0.5, density=True)
    axs[0][1].axvline(x=5, color='black', linestyle='dashed')

    if lib2:
        axs[0][1].hist(library_2['xlogp'], bins=20, alpha=0.5, density=True, color='orange')

    if lib3:
        axs[0][1].hist(library_3['xlogp'], bins=20, alpha=0.5, density=True, color='gray')

    axs[0][1].set_xlabel('XLogP')
    axs[0][1].set_xlim([-6, 7])

    axs[1][0].hist(library_1['tpsa'], bins=20, alpha=0.5, density=True)
    axs[1][0].axvline(x=140, color='black', linestyle='dashed')

    if lib2:
        axs[1][0].hist(library_2['tpsa'], bins=20, alpha=0.5, density=True, color='orange')

    if lib3:
        axs[1][0].hist(library_3['tpsa'], bins=20, alpha=0.5, density=True, color='gray')

    axs[1][0].set_xlabel('TPSA ($\AA$)')
    axs[1][0].set_xlim([20, 200])

    axs[1][1].hist(library_1['hbd'], bins=np.arange(0.5, np.max(library_1['hbd'])+1.5), rwidth=0.7,
                   alpha=0.5, density=True, label=lib1_label)
    axs[1][1].axvline(x=5.3, color='black', linestyle='dashed')

    if lib2:
        axs[1][1].hist(library_2['hbd'], bins=np.arange(0.5, np.max(library_1['hbd'])+1.5), rwidth=0.7,
                   alpha=0.5, density=True, color='orange', label=lib2_label)

    if lib3:
        axs[1][1].hist(library_3['hbd'], bins=np.arange(0.5, np.max(library_1['hbd'])+1.5), rwidth=0.7,
                   alpha=0.5, density=True, color='gray', label=lib3_label)

    axs[1][1].set_xlabel('HBD')
    axs[1][1].set_xlim([-0.5, 6.5])
    axs[1][1].legend(loc='best')

    axs[2][0].hist(np.max(l1p1_sim, axis=1), alpha=0.5, density=True)
    axs[2][0].axvline(x=np.median(np.max(l1p1_sim, axis=1)), linestyle='dashed')

    if lib2:
        axs[2][0].hist(np.max(l2p1_sim, axis=1), alpha=0.5, density=True, color='orange')
        axs[2][0].axvline(x=np.median(np.max(l2p1_sim, axis=1)), linestyle='dashed', color='orange')

    if lib3:
        axs[2][0].hist(np.max(l3p1_sim, axis=1), alpha=0.5, density=True, color='gray')
        axs[2][0].axvline(x=np.median(np.max(l3p1_sim, axis=1)), linestyle='dashed', color='gray')

    axs[2][0].set_xlabel('Nearest Neighbor Tanimoto\nCycle 1')
    axs[2][0].set_xlim([0, 1.0])
    axs[2][0].set_ylim([0, 8])

    axs[2][1].hist(np.max(l1p2_sim, axis=1), alpha=0.5, density=True)
    axs[2][1].axvline(x=np.median(np.max(l1p2_sim, axis=1)), linestyle='dashed')

    if lib2:
        axs[2][1].hist(np.max(l2p2_sim, axis=1), alpha=0.5, density=True, color='orange')
        axs[2][1].axvline(x=np.median(np.max(l2p2_sim, axis=1)), linestyle='dashed', color='orange')


    if lib3:
        axs[2][1].hist(np.max(l3p2_sim, axis=1), alpha=0.5, density=True, color='gray')
        axs[2][1].axvline(x=np.median(np.max(l3p2_sim, axis=1)), linestyle='dashed', color='gray')

    axs[2][1].set_xlabel('Nearest Neighbor Tanimoto\nCycle 2')
    axs[2][1].set_xlim([0, 1.0])
    axs[2][1].set_ylim([0, 8])

    axs[0][0].set_ylabel('Density', fontsize=14)
    axs[1][0].set_ylabel('Density', fontsize=14)
    axs[2][0].set_ylabel('Density', labelpad=10, fontsize=14)


    if lib3:
        axs[3][0].bar(x=-0.5, height=l1p1_cost, width=0.5, alpha=0.5)
        axs[3][0].bar(x=0.5, height=l2p1_cost, width=0.5, alpha=0.5, color='orange')
        axs[3][0].bar(x=1.5, height=l3p1_cost, width=0.5, alpha=0.5, color='gray')
        axs[3][0].set_xticks([-0.5, 0.5, 1.5])
        axs[3][0].set_xticklabels([lib1_label, lib2_label, lib3_label], rotation=30, ha='right')
        axs[3][0].set_xlim([-1, 2])
        axs[3][0].set_ylabel('Cycle 1 BB Cost (USD $)', fontsize=14)

        axs[3][1].bar(x=-0.5, height=l1p2_cost, width=0.5, alpha=0.5)
        axs[3][1].bar(x=0.5, height=l2p2_cost, width=0.5, alpha=0.5, color='orange')
        axs[3][1].bar(x=1.5, height=l3p2_cost, width=0.5, alpha=0.5, color='gray')
        axs[3][1].set_xticks([-0.5, 0.5, 1.5])
        axs[3][1].set_xticklabels([lib1_label, lib2_label, lib3_label], rotation=30, ha='right')
        axs[3][1].set_xlim([-1, 2])
        axs[3][1].set_ylabel('Cycle 2 BB Cost (USD $)', fontsize=14)


    elif lib2:
        axs[3][0].bar(x=-0.25, height=l1p1_cost, width=0.66, alpha=0.5)
        axs[3][0].bar(x=1.25, height=l2p1_cost, width=0.66, alpha=0.5, color='orange')
        axs[3][0].set_xticks([-0.25, 1.25])
        axs[3][0].set_xticklabels([lib1_label, lib2_label], rotation=30, ha='right')
        axs[3][0].set_xlim([-1, 2])
        axs[3][0].set_ylabel('Cycle 1 BB Cost (USD $)', fontsize=14)


        axs[3][1].bar(x=-0.25, height=l1p2_cost, width=0.66, alpha=0.5)
        axs[3][1].bar(x=1.25, height=l2p2_cost, width=0.66, alpha=0.5, color='orange')
        axs[3][1].set_xticks([-0.25, 1.25])
        axs[3][1].set_xticklabels([lib1_label, lib2_label], rotation=30, ha='right')
        axs[3][1].set_xlim([-1, 2])
        axs[3][1].set_ylabel('Cycle 2 BB Cost (USD $)', fontsize=14)

    else:
        axs[3][0].bar(x=0.5, height=l1p1_cost, width=1, alpha=0.5)
        axs[3][0].set_xticks([0.5])
        axs[3][0].set_xticklabels([lib1_label], rotation=30, ha='right')
        axs[3][0].set_xlim([-1, 2])
        axs[3][0].set_ylabel('Cycle 1 BB Cost (USD $)', fontsize=14)

        axs[3][1].bar(x=0.5, height=l1p2_cost, width=1, alpha=0.5)
        axs[3][1].set_xticks([0.5])
        axs[3][1].set_xticklabels([lib1_label], rotation=30, ha='right')
        axs[3][1].set_xlim([-1, 2])
        axs[3][1].set_ylabel('Cycle 2 BB Cost (USD $)', fontsize=14)

    return fig
