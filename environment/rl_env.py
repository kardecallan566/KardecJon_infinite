import numpy as np
import pandas as pd
from typing import List, Dict

class BettingEnv:
    def __init__(self, history_size: int = 20):
        self.history_size = history_size
        self.color_map = {"PRETO": 0, "VERDE": 1, "BRANCO": 2}
        self.inv_color_map = {v: k for k, v in self.color_map.items()}

    def get_state(self, history_df: pd.DataFrame) -> np.ndarray:
        """
        Converts history dataframe into a feature vector.
        """
        if len(history_df) < self.history_size:
            # Pad with zeros if history is too short
            rows = history_df.to_dict('records')
            while len(rows) < self.history_size:
                rows.append({"color": "BRANCO", "number": 0})
            history_df = pd.DataFrame(rows)
        
        recent = history_df.head(self.history_size)
        
        # Features:
        # 1. Color history (one-hot or normalized)
        colors = [self.color_map.get(c, 2) for c in recent['color']]
        
        # 2. Color frequencies
        counts = recent['color'].value_counts(normalize=True).to_dict()
        freq_preto = counts.get("PRETO", 0)
        freq_verde = counts.get("VERDE", 0)
        freq_branco = counts.get("BRANCO", 0)
        
        # 3. Sequence lengths (how many consecutive same colors)
        sequences = []
        current_seq = 1
        color_list = recent['color'].tolist()
        for i in range(1, len(color_list)):
            if color_list[i] == color_list[i-1]:
                current_seq += 1
            else:
                sequences.append(current_seq)
                current_seq = 1
        sequences.append(current_seq)
        avg_seq = np.mean(sequences)
        max_seq = np.max(sequences)
        
        # 4. Alternation rate
        alternations = sum(1 for i in range(1, len(color_list)) if color_list[i] != color_list[i-1])
        alt_rate = alternations / (self.history_size - 1)
        
        # 5. Last number
        last_num = recent.iloc[0]['number'] / 14.0
        
        # Construct feature vector
        features = colors + [freq_preto, freq_verde, freq_branco, avg_seq, max_seq, alt_rate, last_num]
        return np.array(features, dtype=np.float32)

    def calculate_reward(self, prediction: str, actual: str, consecutive_wins: int) -> float:
        if prediction == actual:
            reward = 1.0
            # Bonus for consecutive wins
            if consecutive_wins > 1:
                reward += 0.1 * min(consecutive_wins, 5)
            return reward
        else:
            return -1.0
