from peptide_pipeline.chemist.base import BaseChemist 
from typing import List, Dict, Any
import peptides as pp
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor, SDWriter

class ChemistAgent(BaseChemist):
    def __init__(self, min_length: int = 2, max_length: int = 10):
        super().__init__()
        self.logger = get_logger("ChemistAgent")
        self.basics_aa = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
        self.MIN_LENGTH = min_length
        self.MAX_LENGTH = max_length

    def check_validity(self, peptides: List[str]) -> List[bool]:
        """
        Check if peptide sequence is valid.
        Returns a list of booleans (True if valid, False otherwise).
        """
        results = []

        if not peptides:
            self.logger.error("Invalid format: Empty peptide list provided")
            return []

        for id, seq in enumerate(peptides):

            #empty sequence
            if not seq:
                self.logger.warning(f"Sequence ID {id} : Invalid peptide: Empty sequence found")
                results.append(False)
                continue
            
            #check length
            if len(seq) < self.MIN_LENGTH or len(seq) > self.MAX_LENGTH:
                self.logger.warning(f"Sequence ID {id} : Bad sequence size for {seq}")
                results.append(False)
                continue
            
            #only contains basics amino-acids (convert in caps first)
            upper_seq = seq.upper()
            if not all(char in self.basics_aa for char in upper_seq):
                invalid_chars = set(upper_seq) - self.basics_aa
                self.logger.warning(f"Sequence ID {id} : Invalid peptide '{seq}': contains invalid amino-acids {invalid_chars}")
                results.append(False)
                continue

            #check constructible and molecule Sanitization from sequence

            mol = Chem.MolFromSequence(seq)
            
            if mol is None:
                self.logger.warning(f"Sequence ID {id} : Peptide not constructible from sequence {seq}")
                results.append(False)
                continue

            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                self.logger.warning(f"Sequence ID {id} : Sanitization Failed,  Chemical physics violation for '{seq}': {e}")
                results.append(False)
                continue

            results.append(True)
        
        valid_count = sum(results)
        self.logger.info(f"Validity check complete: {valid_count}/{len(peptides)} valid peptides")    
        return results

    def calculate_properties(self, peptides_list: List[str]) -> List[Dict[str, float]]:
        """
        Calculates physicochemical properties for a list of short peptides.
        """
        props_list = []

        for seq in peptides_list:
            try:
                pep = pp.Peptide(seq)
                mol = Chem.MolFromSequence(seq)
                
                if mol is None:
                    raise ValueError("RDKit could not build the molecule.")

                props = {
                    "size": len(seq),
                    "molecular_weight": pep.molecular_weight(),
                    
            
                    "net_charge_pH5_5": pep.charge(pH=5.5), 
                    "isoelectric_point": pep.isoelectric_point(),
                    
                    "hydrophobicity": pep.hydrophobicity(),
                    "hydrophobic_moment" : pep.hydrophobic_moment(),
                    "logp": Descriptors.MolLogP(mol),
                    
                    "boman_index": pep.boman(),
                    
                    "h_bond_donors": Descriptors.NumHDonors(mol),
                    "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                    "tpsa": Descriptors.TPSA(mol) 
                }
                props_list.append(props)

            except Exception as e:
                self.logger.warning(f"Failed to calculate properties for '{seq}': {str(e)}")
                props_list.append({})
        
        return props_list

    def filter_peptides(self, peptides: List[str], constraints: Dict[str, Any]) -> List[str]:
            """
            Filters a list of peptides based on chemical constraints.
            Example of constraints format:
            {
                "size": {"min": 2, "max": 10},
                "net_charge": {"min": 1.0},
                "logp": {"max": 5.0}
            }
            """
            if not constraints:
                self.logger.info("No constraints provided. Returning original list.")
                return peptides

            properties_list = self.calculate_properties(peptides)
            filtered_peptides = []

            for seq, props in zip(peptides, properties_list):
                if not props:
                    self.logger.debug(f"Excluded '{seq}': Missing properties.")
                    continue

                is_valid = True

                for prop_name, limits in constraints.items():
                    if prop_name not in props:
                        self.logger.warning(f"Unknown constraint '{prop_name}' ignored.")
                        continue

                    value = props[prop_name]

                    if "min" in limits and value < limits["min"]:
                        is_valid = False
                        break 
                    
                    if "max" in limits and value > limits["max"]:
                        is_valid = False
                        break

                if is_valid:
                    filtered_peptides.append(seq)

            self.logger.info(f"Filtering complete: {len(filtered_peptides)}/{len(peptides)} peptides kept.")
            return filtered_peptides
    
    def create_sdf_file(self, peptides: List[str], path: str):
        """
        Computes a .sdf file for each peptide in the list
        .sdf includes 2DCoords and properties computed by the calculate_properties() function
        """
        props_list = self.calculate_properties(peptides)
        with SDWriter(path) as writer:
            for seq, props in zip(peptides, props_list):
                mol = Chem.MolFromSequence(seq)
                if mol is None:
                    self.logger.warning(f"Peptide not constructible from sequence {seq}")
                    continue
                rdDepictor.Compute2DCoords(mol)
                
                for key, value in props.items():
                    mol.SetProp(key, str(value))
                
                writer.write(mol)







