# files

## What's here:
- `pamine_df.csv`: cleaned file with strictly primary amines and truncates for each BB
- `pAA_df.csv`: cleaned file with strictly primary amino acids and truncates for each BB
- `cooh_df.csv`: cleaned file with strictly carboxylic acids and truncates for each BB
- `all_truncates.csv`: file with concatenated list of unique truncates across all BB sets
- `truncates_UMAP.csv`: file with UMAP coordinates for each unique truncate

#### stock_files
- `pamine_stock.sdf`: catalog file of amine building blocks sourced from Enamine
- `AA_stock.sdf`: catalog file of amino acid building blocks sourced from Enamine
- `cooh_stock.sdf`: catalog file of carboxylic acid building blocks sourced from Enamine

- `pamine_w_truncates.sdf`: catalog file with truncated versions of strictly primary amine building blocks
- `pAA_w_truncates.sdf`: catalog file with truncated versions of strictly primary amino acid building blocks
- `cooh_w_truncates.sdf`: catalog file with truncated versions of carboxylic acid building blocks

#### library_enum_lists
- `pamine_{selection}_{size}.csv`: list of {size} primary amine BBs chosen using a {selection} strategy
- `cooh_{selection}_{size}.csv`: list of {size} carboxylic acid BBs chosen using a {selection} strategy 
