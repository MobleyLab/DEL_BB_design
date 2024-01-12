# code

## What's here:
#### data_prep
- `clean_data.ipynb`: notebook to clean Enamine BB catalog files for analysis
- `deprot_SMIRKS.pkl`: pickled file of dictionary containing SMIRKS for removing certain protecting groups
- `PG_SMILES.pkl`: pickled file of dictionary containing SMILES patterns for certain protecting groups
- `run_2D_dist.py`: calculates the chemical distance (1 - 2D Tanimoto similarity) for a given batch of compounds
- `umap_projection.py`: generates UMAP coordinates for compounds

#### analysis
- `analysis.ipynb`: notebook to reproduce all results and figures reported in the manuscript 
- `SI_analysis.ipynb`: notebook to reproduce all results and figures reported in the Supporting Information

#### library_enum
- `custom_library_enumeration.ipynb`: notebook to simulate various library designs
- `helper.py`: Python script containing various helper functions

