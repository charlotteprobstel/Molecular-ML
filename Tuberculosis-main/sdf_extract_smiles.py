from rdkit import Chem
import pandas as pd

def extract_molecule_properties(sdf_path, smiles='Smiles', inchi='InChi', activity='Activity -logM'):
    """
    Extracts SMILES, InChi, and Activity from an SDF file.

    Args:
    sdf_path (str): Path to the SDF file.

    Returns:
    list of dicts: Each dictionary contains SMILES, InChi, and Activity for a molecule.
    """
    supplier = Chem.SDMolSupplier(sdf_path)
    results = []

    # Iterate over all molecules in the SDF file
    for mol in supplier:
        if mol is not None:  # Check if the molecule could be read successfully
            # Extract properties
            data = {
                'SMILES': mol.GetProp(smiles) if mol.HasProp(smiles) else 'N/A',
                'InChi': mol.GetProp(inchi) if mol.HasProp(inchi) else 'N/A',
                'Activity': mol.GetProp(activity) if mol.HasProp(activity) else 'N/A'
            }
            results.append(data)

    return pd.DataFrame(results)

# Example function call (commented out):

path='/rds/general/user/qg622/home/Y2/Antibacterial/data/2017_Mtb_testset.sdf'
molecule_data = extract_molecule_properties(path)
molecule_data.to_csv(f'{path[:-3]}csv', index=False)

path='/rds/general/user/qg622/home/Y2/Antibacterial/data/Mtb_predictions_Vadim_100nM.sdf'
molecule_data = extract_molecule_properties(path, inchi='Inchi', activity='-log(M)_valuemeanmean')
molecule_data.to_csv(f'{path[:-3]}csv', index=False)

path='/rds/general/user/qg622/home/Y2/Antibacterial/data/Mtb_published_regression_AC_Cleaned.sdf'
molecule_data = extract_molecule_properties(path, smiles='SMILES', inchi='InChI', activity='Actiivty -logM')
molecule_data.to_csv(f'{path[:-3]}csv', index=False)

path='/rds/general/user/qg622/home/Y2/Antibacterial/data/RCB_Mtb_inhibition_2007-2019.sdf'
molecule_data = extract_molecule_properties(path, smiles='SMILES', inchi='InChI', activity='Activity -Log[M]')
molecule_data.to_csv(f'{path[:-3]}csv', index=False)


