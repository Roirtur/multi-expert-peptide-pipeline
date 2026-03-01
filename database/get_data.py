import requests
import json
import time

def extract_generative_ai_dataset():
    SEARCH_URL = "https://dbaasp.org/peptides"
    
    # INITIAL SEARCH
    search_params = {
        'sequenceLength.value': '5-12',
        'targetGroup.value': 'Gram+',     # antibacterial activity
        'complexity.value': 'monomer',    # Keeps the structure simple
        'limit': 5
    }
    
    print("Step 1: Searching for peptides...")
    search_res = requests.get(SEARCH_URL, params=search_params, headers={'accept': 'application/json'})
    
    if search_res.status_code != 200:
        print(f"Error fetching search list: {search_res.status_code}")
        return []

    try:
        res_json = search_res.json()
        data = res_json.get('data', [])
        print("Exists : ", res_json.get("totalCount"))
        print("Found :", len(data))
    except:
        raise FileNotFoundError
    
    peptides_list = data
    final_data = []

    print(f"Step 2: Fetching stats for {len(peptides_list)} peptides...")
    
    for entry in peptides_list:
        p_id = entry.get('dbaaspId') or entry.get('id')
        detail_url = f"https://dbaasp.org/peptides/{p_id}"
        detail_res = requests.get(detail_url, headers={'accept': 'application/json'})
        
        if detail_res.status_code == 200:
            full_info = detail_res.json()
            
            raw_props = full_info.get('physicoChemicalProperties') or []
            props = {p.get('name'): p.get('value') for p in raw_props if p.get('name')}
            
            raw_activities = full_info.get("targetActivities") or []
            parsed_activities = []
            for act in raw_activities:
                parsed_activities.append({
                    "species": (act.get("targetSpecies") or {}).get("name"),
                    "measure": act.get("activityMeasureValue"),
                    "concentration": act.get("concentration"),
                    "activity_value": act.get("activity"),
                    "unit": (act.get("unit") or {}).get("name")
                })

            raw_smiles = full_info.get("smiles") or []
            smiles_list = [s.get("smiles") for s in raw_smiles if s.get("smiles")]

            raw_target_groups = full_info.get("targetGroups") or []
            target_groups = [(tg or {}).get("name") for tg in raw_target_groups]

            raw_unusual_aa = full_info.get("unusualAminoAcids") or []
            unusual_aa = [
                {
                    "position": uaa.get("position"),
                    "modification": (uaa.get("modificationType") or {}).get("name")
                } 
                for uaa in raw_unusual_aa
            ]

            refined_entry = {
                "dbaasp_id": full_info.get("dbaaspId"),
                "sequence": full_info.get("sequence"),
                "length": full_info.get("sequenceLength"),
                "complexity": (full_info.get("complexity") or {}).get("name"),
                "synthesis_type": (full_info.get("synthesisType") or {}).get("name"),
                
                # Structural Features
                "smiles": smiles_list,
                "unusual_amino_acids": unusual_aa,
                "net_charge": float(props.get("Net Charge", 0)),
                "isoelectric_point": float(props.get("Isoelectric Point", 0)),
                "hydrophobicity": float(props.get("Normalized Hydrophobicity", 0)),
                "ppii_propensity": float(props.get("Propensity to PPII coil", 0)),
                "amphiphilicity_index": float(props.get("Amphiphilicity Index", 0)),
                
                # Target Features
                "target_groups": target_groups,
                "activities": parsed_activities
            }
            
            final_data.append(refined_entry)
            print(f" Saved {refined_entry['dbaasp_id']} | Seq: {refined_entry['sequence']} | SMILES found: {len(smiles_list)}")
            
            # time.sleep(0.5)

    return final_data

dataset = extract_generative_ai_dataset()
with open('ai_training_peptides.json', 'w') as f:
    json.dump(dataset, f, indent=4)
print("\nDataset successfully saved to ai_training_peptides.json")