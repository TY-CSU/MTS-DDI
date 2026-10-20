import xml.etree.ElementTree as ET
import pandas as pd
import os
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def parse_drugbank_xml(xml_file, output_csv):
    print(f"Parsing XML file: {xml_file}")
    
    # Namespace map
    ns = {'db': 'http://www.drugbank.ca'}
    
    data = []
    
    # Iterative parsing to handle large files
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in tqdm(context, desc="Parsing Drugs"):
        if elem.tag == '{http://www.drugbank.ca}drug':
            drug_id = None
            smiles = None
            
            # Get DrugBank ID (Primary)
            for id_elem in elem.findall('db:drugbank-id', ns):
                if id_elem.get('primary') == 'true':
                    drug_id = id_elem.text
                    break
            
            # Get SMILES
            # Structure: calculated-properties -> property -> kind: SMILES, value: ...
            calc_props = elem.find('db:calculated-properties', ns)
            if calc_props is not None:
                for prop in calc_props.findall('db:property', ns):
                    kind = prop.find('db:kind', ns)
                    if kind is not None and kind.text == 'SMILES':
                        value = prop.find('db:value', ns)
                        if value is not None:
                            smiles = value.text
                            break
            
            if drug_id and smiles:
                data.append({'DrugBank_ID': drug_id, 'Smiles': smiles})
            
            # Clear element to save memory
            elem.clear()
            
    df = pd.DataFrame(data)
    print(f"Extracted {len(df)} SMILES from DrugBank XML.")
    
    df.to_csv(output_csv, index=False)
    print(f"Saved mapping to {output_csv}")

if __name__ == "__main__":
    xml_path = os.path.join(Config.RAW_DATA_DIR, 'full database.xml')
    output_path = os.path.join(Config.PROCESSED_DATA_DIR, 'drugbank_smiles_mapping.csv')
    
    if os.path.exists(xml_path):
        parse_drugbank_xml(xml_path, output_path)
    else:
        print(f"Error: XML file not found at {xml_path}")
