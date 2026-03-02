import requests
import json
import time
import argparse
import os

def extract_generative_ai_dataset(limit=5000, batch_size=1000, output_file='ai_training_peptides.json', checkpoint_interval=500):
    SEARCH_URL = "https://dbaasp.org/peptides"
    
    final_data = []
    total_fetched = 0
    offset = 0
    
    print(f"Goal: Fetch {limit} peptides (max batch size: {batch_size}). Saving every {checkpoint_interval} entries.")

    while total_fetched < limit:
        current_request_limit = min(batch_size, limit - total_fetched)
        
        search_params = {
            'sequenceLength.value': '5-12',
            'targetGroup.value': 'Gram+',     # antibacterial activity
            'complexity.value': 'monomer',    # Keeps the structure simple
            'limit': current_request_limit,
            'offset': offset
        }
        
        print(f"\n[Batch Start] Offset: {offset}, Requesting: {current_request_limit}...")
        
        try:
            search_res = requests.get(SEARCH_URL, params=search_params, headers={'accept': 'application/json'})
            
            if search_res.status_code != 200:
                print(f"Error fetching search list: {search_res.status_code}")
                break
                
            res_json = search_res.json()
            data = res_json.get('data', [])
            total_available = res_json.get("totalCount", 0)
            
            if not data:
                print("No more data returned from API.")
                break
                
        except Exception as e:
            print(f"Exception during search: {e}")
            break
        
        peptides_list = data

        print(f"\nGot {len(peptides_list)} peptides for this batch.")
        print(f"\nFetching details...")
        
        # Fetch details for each peptide
        for i, entry in enumerate(peptides_list):
            p_id = entry.get('dbaaspId') or entry.get('id')
            detail_url = f"https://dbaasp.org/peptides/{p_id}"
            
            try:
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
                    current_count = len(final_data)
                    
                    if current_count % checkpoint_interval == 0:
                        with open(output_file, 'w') as f:
                            json.dump(final_data, f, indent=4)
                        print(f"\n--- [{current_count}/{limit}] Checkpoint: Saved {current_count} entries to {output_file} ---")

            except Exception as e:
                print(f"\nError processing {p_id}: {e}")
                
            # No sleep needed if we want speed, or small sleep
            # time.sleep(0.01)

        offset += len(data)
        total_fetched += len(data)
        
        if len(data) < current_request_limit:
             print("\nEnd of available data reached.")
             break

    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch peptide data from DBAASP")
    parser.add_argument("--limit", type=int, default=5000, help="Total number of peptides to fetch (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Max items per request (default: 1000)")
    parser.add_argument("--output", type=str, default="ai_training_peptides.json", help="Output JSON file")
    parser.add_argument("--checkpoint", type=int, default=500, help="Save every N entries (default: 500)")

    args = parser.parse_args()

    dataset = extract_generative_ai_dataset(
        limit=args.limit,
        batch_size=args.batch_size,
        output_file=args.output,
        checkpoint_interval=args.checkpoint
    )
    
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"\nFinal dataset successfully saved to {args.output} ({len(dataset)} items)")
