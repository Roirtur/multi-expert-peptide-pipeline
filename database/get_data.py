import aiohttp
import asyncio
import json
import time
import argparse
import os

MAX_CONCURRENT_REQUESTS = 20

async def fetch_detail(session, p_id):
    detail_url = f"https://dbaasp.org/peptides/{p_id}"
    try:
        async with session.get(detail_url, headers={'accept': 'application/json'}) as response:
            if response.status == 200:
                try:
                    return await response.json()
                except Exception:
                    return None
            return None
    except Exception as e:
        print(f"Error fetching {p_id}: {e}")
        return None

def process_peptide_data(full_info):
    if not full_info:
        return None

    sequence = full_info.get("sequence")
    
    raw_smiles = full_info.get("smiles") or []
    smiles_list = [s.get("smiles") for s in raw_smiles if s.get("smiles")]
    
    if not sequence and not smiles_list:
        return None

    raw_props = full_info.get('physicoChemicalProperties') or []
    props = {p.get('name'): p.get('value') for p in raw_props if p.get('name')}

    raw_target_groups = full_info.get("targetGroups") or []
    target_groups = [(tg or {}).get("name") for tg in raw_target_groups]

    refined_entry = {
        "dbaasp_id": full_info.get("dbaaspId"),
        "sequence": sequence,
        "length": full_info.get("sequenceLength"),
        "smiles": smiles_list[0] if smiles_list else None, 
        "all_smiles": smiles_list,
        "target_groups": target_groups,
        "complexity": full_info.get("complexity"),
        "net_charge": float(props.get("Net Charge", 0)),
        "isoelectric_point": float(props.get("Isoelectric Point", 0)),
        "hydrophobicity": float(props.get("Hydrophobicity", 0)),
        "normalized_hydrophobicity": float(props.get("Normalized Hydrophobicity", 0)),
        "ppii_propensity": float(props.get("PPII Propensity", 0)), 
        "amphiphilicity_index": float(props.get("Amphiphilicity Index", 0)),
    }
    return refined_entry

async def fetch_batch_details(session, peptides_list):
    tasks = []
    for entry in peptides_list:
        p_id = entry.get('dbaaspId') or entry.get('id')
        tasks.append(fetch_detail(session, p_id))
    
    results = await asyncio.gather(*tasks)
    return results

async def extract_generative_ai_dataset_async(limit=5000, batch_size=1000, output_file='ai_training_peptides.json'):
    SEARCH_URL = "https://dbaasp.org/peptides"
    
    final_data = []
    total_fetched = 0
    offset = 0
    
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"Goal: Fetch {limit} peptides. Async mode active.")

        while total_fetched < limit:
            current_request_limit = min(batch_size, limit - total_fetched)
            
            # Using antibacterial search criteria
            search_params = {
                'sequenceLength.value': '2-12',   
                'targetGroup.value': 'Gram+',     
                'complexity.value': 'monomer',    
                'limit': current_request_limit,
                'offset': offset
            }
            
            print(f"\n[Batch Start] Offset: {offset}, Search Request: {current_request_limit}...")
            
            try:
                async with session.get(SEARCH_URL, params=search_params, headers={'accept': 'application/json'}) as resp:
                    if resp.status != 200:
                        print(f"Error fetching search list: {resp.status}")
                        break
                    res_json = await resp.json()
            except Exception as e:
                print(f"Search Exception: {e}")
                break
                
            data = res_json.get('data', [])
            if not data:
                print("No more data found.")
                break

            print(f"  > Found {len(data)} peptides. Fetching details concurrently...")
            
            details_raw = await fetch_batch_details(session, data)
            
            valid_entries = []
            for d in details_raw:
                processed = process_peptide_data(d)
                if processed:
                    valid_entries.append(processed)
            
            final_data.extend(valid_entries)
            
            if len(final_data) > 0:
                 with open(output_file, 'w') as f:
                    json.dump(final_data, f, indent=4)
                 print(f"  > Saved {len(final_data)} total entries so far.")

            offset += len(data)
            total_fetched += len(data)
            
            if len(data) < current_request_limit:
                print("End of search results.")
                break
                
    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch peptide data (Async)")
    parser.add_argument("--limit", type=int, default=5000, help="Total number of peptides to fetch")
    parser.add_argument("--batch-size", type=int, default=1000, help="Search batch size")
    parser.add_argument("--output", type=str, default="ai_training_peptides.json", help="Output JSON file")

    args = parser.parse_args()

    start_time = time.time()
    result = asyncio.run(extract_generative_ai_dataset_async(
        limit=args.limit,
        batch_size=args.batch_size,
        output_file=args.output
    ))
    end_time = time.time()
    
    print(f"\nCompleted! Collected {len(result)} peptides in {end_time - start_time:.2f} seconds.")
