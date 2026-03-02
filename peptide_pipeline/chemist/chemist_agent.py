from peptide_pipeline.chemist import BaseChemist 
from typing import List, Dict, Any
from logger.logger import get_logger
import peptides as pp
from rdkit import Chem
from rdkit.Chem import Descriptors

class ChemistAgent(BaseChemist):
    def __init__(self):
        super().__init__()
        # On initialise un logger spécifique à cet agent
        self.logger = get_logger("ChemistAgent")

    def check_validity(self, peptides: List[str]) -> List[bool]:
        """
        Check if peptide sequence is empty or contains abnormal amino-acids.
        Returns a list of booleans (True if valid, False otherwise).
        """
        basics_aa = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
        results = []
        
        if not peptides:
            self.logger.error("Invalid format: Empty peptide list provided")
            return []

        for seq in peptides:
            if not seq:
                self.logger.warning("Invalid peptide: Empty sequence found")
                results.append(False)
                continue

            
            upper_seq = seq.upper()
            if all(char in basics_aa for char in upper_seq):
                self.logger.debug(f"Valid peptide sequence: {seq}")
                results.append(True)
            else:
                invalid_chars = set(upper_seq) - basics_aa
                self.logger.warning(f"Invalid peptide '{seq}': contains invalid amino-acids {invalid_chars}")
                results.append(False)
        
        valid_count = sum(results)
        self.logger.info(f"Validity check complete: {valid_count}/{len(peptides)} valid peptides")    
        return results

    def calculate_properties(self, peptides_list: List[str]) -> List[Dict[str, float]]:

        """
        Calculates physicochemical properties for a list of peptides:
        """

        props_list = []

        for seq in peptides_list:

            try:
                pep = pp.Peptide(seq)
                props = {
                    "size" :  len(seq),
                    "molecular_weight" : pep.molecular_weight(),
                    "net_charge" : pep.charge(pH = 5.5), #physiological ph might change later
                    "hydrophobicite" : pep.hydrophobicity(),
                    #get logp with rdkit
                    "logp" : Descriptors.MolLogP(Chem.MolFromSequence(seq)),
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







