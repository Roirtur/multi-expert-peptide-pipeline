import aiohttp
import asyncio
import json
import time
import argparse
import os

# Import RDKit for calculating Molecular Weight and LogP from SMILES
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not installed. Molecular Weight and LogP calculations will be skipped.")
    print("To install RDKit, run: pip install rdkit")

MAX_CONCURRENT_REQUESTS = 150  # Increased for faster concurrent fetching
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY") # Pre-compiled to save loop overhead

async def fetch_detail(session, p_id):
    """Fetch detailed peptide information from DBAASP."""
    detail_url = f"https://dbaasp.org/peptides/{p_id}"
    try:
        # Added a timeout to prevent hanging requests from stalling the batch
        async with session.get(detail_url, headers={'accept': 'application/json'}) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    return data, None
                except Exception as e:
                    return None, f"JSONDecodeError: {type(e).__name__}"
            return None, f"HTTP {response.status}"
    except Exception as e:
        return None, type(e).__name__

def process_peptide_data(full_info):
    """Process and extract/calculate the requested properties for conditioning."""
    if not full_info:
        return None

    sequence = full_info.get("sequence")
    
    # Fix and validate sequence (convert to uppercase, filter out invalid characters)
    if sequence:
        sequence = sequence.upper()
        if not set(sequence).issubset(VALID_AMINO_ACIDS):
            # Filter out entries with non-standard or unknown characters
            return None

    # Extract SMILES
    raw_smiles = full_info.get("smiles") or []
    smiles_list = [s.get("smiles") for s in raw_smiles if s.get("smiles")]
    primary_smiles = smiles_list[0] if smiles_list else None
    
    if not sequence and not primary_smiles:
        return None

    # Parse physicochemical properties into a fast-lookup dictionary
    raw_props = full_info.get('physicoChemicalProperties') or []
    props = {p.get('name'): p.get('value') for p in raw_props if p.get('name')}

    # Extract pH from target activities (taking average if multiple distinct pH tests exist)
    target_activities = full_info.get("targetActivities") or []
    phs = []
    for act in target_activities:
        ph_val = act.get("ph")
        if ph_val is not None:
            try:
                phs.append(float(ph_val))
            except ValueError:
                pass
    avg_ph = sum(phs) / len(phs) if phs else None

    # Calculate Molecular Weight and LogP using RDKit
    molecular_weight = None
    logp = None
    if RDKIT_AVAILABLE and primary_smiles:
        try:
            mol = Chem.MolFromSmiles(primary_smiles)
            if mol:
                molecular_weight = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
        except Exception:
            pass # Failsafe for invalid SMILES strings
            
    # Calculate Cationicity (number of positively charged basic residues)
    cationicity = 0
    if sequence:
        cationicity = sequence.count('K') + sequence.count('R') + sequence.count('H')

    # Helper function to safely parse float values from the API
    def safe_float(val, default=None):
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Extract target groups
    raw_target_groups = full_info.get("targetGroups") or []
    target_groups = [(tg or {}).get("name") for tg in raw_target_groups]
    
    # Complexity
    complexity = full_info.get("complexity")
    complexity_name = complexity.get("name") if complexity else None

    # Final refined entry formatting according to requirements
    refined_entry = {
        "dbaasp_id": full_info.get("dbaaspId"),
        "sequence": sequence,
        "length": full_info.get("sequenceLength"),
        "smiles": primary_smiles,
        
        # --- Target Conditioning Features ---
        "ph": avg_ph,
        "molecular_weight": molecular_weight,
        "logp": logp,
        "net_charge": safe_float(props.get("Net Charge")),
        "isoelectric_point": safe_float(props.get("Isoelectric Point")),
        "hydrophobicity": safe_float(props.get("Normalized Hydrophobicity")),
        "cathionicity": cationicity,
        # ------------------------------------
        
        "target_groups": target_groups,
        "complexity": complexity_name,
    }
    
    return refined_entry

async def fetch_batch_details(session, peptides_list):
    """Fetch details concurrently for a batch of peptides."""
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
            
            # Using antibacterial search criteria (customizable)
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
            
            details_with_errors = await fetch_batch_details(session, data)
            
            valid_entries = []
            batch_errors = {}
            
            for d, err in details_with_errors:
                if err:
                    batch_errors[err] = batch_errors.get(err, 0) + 1
                
                processed = process_peptide_data(d)
                if processed:
                    valid_entries.append(processed)
            
            final_data.extend(valid_entries)
            
            print(f"  > Processed and added {len(valid_entries)} valid entries from this batch.")
            if batch_errors:
                print(f"  > Batch Errors Summary: {batch_errors}")

            offset += len(data)
            total_fetched += len(data)
            
            if len(data) < current_request_limit:
                print("End of search results.")
                break
                
        # Save exactly once at the end to drastically reduce I/O overhead
        if final_data:
             with open(output_file, 'w') as f:
                json.dump(final_data, f, indent=4)
             print(f"\nSuccessfully saved {len(final_data)} total entries to {output_file}.")

    return final_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch peptide data with physicochemical calculations (Async)")
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
    
    print(f"\nCompleted! Collected and processed {len(result)} peptides in {end_time - start_time:.2f} seconds.")