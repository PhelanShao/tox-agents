
# ================================================================================
# ToxD4C Confidence Evaluation Usage Example
# ================================================================================

import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class ToxD4CConfidenceEvaluator:
    """ToxD4C model prediction confidence evaluator"""
    
    def __init__(self, data_dir):
        """
        Initialize evaluator
        
        Args:
            data_dir: ToxD4C data directory path
        """
        # Load training set fingerprints
        fp_data = np.load(f"{data_dir}/train_fingerprints.npz", allow_pickle=True)
        self.train_fps_array = fp_data['fingerprints']
        self.train_smiles = fp_data['smiles'].tolist()
        
        # Convert to RDKit fingerprint objects (for fast similarity calculation)
        self.train_fps = []
        for fp_array in self.train_fps_array:
            fp = DataStructs.ExplicitBitVect(2048)
            for i, bit in enumerate(fp_array):
                if bit:
                    fp.SetBit(i)
            self.train_fps.append(fp)
        
        # Load configuration
        with open(f"{data_dir}/confidence_config.pkl", 'rb') as f:
            self.config = pickle.load(f)
        
        self.thresholds = self.config['thresholds']
        print(f"Loaded {len(self.train_fps)} training set fingerprints")
    
    def get_morgan_fp(self, smiles):
        """Generate Morgan fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except:
            return None
    
    def calc_max_similarity(self, query_fp):
        """Compute maximum similarity between query molecule and training set"""
        max_sim = 0.0
        for train_fp in self.train_fps:
            sim = DataStructs.TanimotoSimilarity(query_fp, train_fp)
            if sim > max_sim:
                max_sim = sim
        return max_sim
    
    def get_tanimoto_confidence(self, similarity):
        """Return confidence level based on Tanimoto similarity"""
        t = self.thresholds['tanimoto']
        if similarity >= t['high']:
            return 'High', 'High confidence'
        elif similarity >= t['medium_high']:
            return 'Medium-High', 'Medium-high confidence'
        elif similarity >= t['medium']:
            return 'Medium', 'Medium confidence'
        elif similarity >= t['low']:
            return 'Low', 'Low confidence'
        else:
            return 'Out_of_Domain', 'Out of domain'
    
    def get_probability_confidence(self, prob):
        """Return confidence level based on prediction probability"""
        t = self.thresholds['probability']
        if prob >= t['definite_positive']:
            return 'Definite_Positive', 'Definite positive', 'Toxic'
        elif prob >= t['probable_positive']:
            return 'Probable_Positive', 'Probable positive', 'Possibly_Toxic'
        elif prob >= t['uncertain']:
            return 'Uncertain', 'Uncertain', 'Unknown'
        elif prob >= t['probable_negative']:
            return 'Probable_Negative', 'Probable negative', 'Possibly_Safe'
        else:
            return 'Definite_Negative', 'Definite negative', 'Safe'
    
    def get_combined_grade(self, similarity, prob):
        """Get combined confidence level"""
        prob_distance = abs(prob - 0.5)
        
        if similarity >= 0.70 and prob_distance >= 0.30:
            return 'A', 'High reliability', 'Use directly'
        elif similarity >= 0.50 and prob_distance >= 0.20:
            return 'B', 'Reliable', 'Suggest review'
        elif similarity >= 0.38 and prob_distance >= 0.10:
            return 'C', 'Use with caution', 'Need validation'
        else:
            return 'D', 'Unreliable', 'Do not use'
    
    def evaluate(self, smiles, pred_prob):
        """
        Evaluate prediction confidence for a single molecule
        
        Args:
            smiles: Molecule SMILES
            pred_prob: Model predicted toxicity probability
            
        Returns:
            dict: Dictionary containing confidence metrics
        """
        fp = self.get_morgan_fp(smiles)
        if fp is None:
            return {
                'smiles': smiles,
                'error': 'Invalid SMILES',
                'max_similarity': None,
                'confidence_grade': 'D'
            }
        
        max_sim = self.calc_max_similarity(fp)
        tani_conf = self.get_tanimoto_confidence(max_sim)
        prob_conf = self.get_probability_confidence(pred_prob)
        combined = self.get_combined_grade(max_sim, pred_prob)
        
        return {
            'smiles': smiles,
            'pred_probability': pred_prob,
            'max_similarity': max_sim,
            'tanimoto_confidence': tani_conf[0],
            'tanimoto_confidence_cn': tani_conf[1],
            'probability_confidence': prob_conf[0],
            'probability_confidence_cn': prob_conf[1],
            'toxicity_judgment': prob_conf[2],
            'combined_grade': combined[0],
            'combined_grade_cn': combined[1],
            'recommendation': combined[2]
        }
    
    def evaluate_batch(self, smiles_list, pred_probs):
        """Batch evaluation"""
        results = []
        for smi, prob in zip(smiles_list, pred_probs):
            results.append(self.evaluate(smi, prob))
        return results


# ================================================================================
# Usage example
# ================================================================================
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ToxD4CConfidenceEvaluator(
        data_dir="/path/to/D4C-new/ToxD4C_en/data"
    )
    
    # Single molecule evaluation
    result = evaluator.evaluate(
        smiles="CCO",  # Ethanol
        pred_prob=0.25  # Model predicted toxicity probability
    )
    
    print("Evaluation results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Batch evaluation
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
    pred_probs = [0.25, 0.75, 0.45]
    
    results = evaluator.evaluate_batch(smiles_list, pred_probs)
    for r in results:
        print(f"\n{r['smiles']}: Grade {r['combined_grade']} ({r['combined_grade_cn']})")
