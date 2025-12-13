import pandas as pd
import numpy as np

# --- 1. Helper Functions (MUST be defined before compute_features) ---

def get_distance(x1, y1, x2, y2):
    """Euclidean distance between two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_angle(x1, y1, x2, y2):
    """Calculates the angle (in radians) of the pass relative to the field."""
    return np.arctan2(y2 - y1, x2 - x1)

def is_same_team(sender_id, player_id):
    """
    Returns 1 if sender and player are on the same team, 0 otherwise.
    Teams: 1-11 (Home), 12-22 (Away).
    """
    if sender_id <= 11:
        return 1 if player_id <= 11 else 0
    else:
        return 1 if player_id > 11 else 0

# --- 2. Main Feature Computation Function ---

def compute_features(X_df, y_df=None):
    """
    Transforms raw data into a candidate-receiver dataset.
    Includes 'Pressure' (dist to nearest opponent) and 'Ranks'.
    """
    n_samples = X_df.shape[0]
    data = []
    
    # Convert to list of dicts for speed (faster than iloc in loop)
    X_records = X_df.to_dict('records')
    y_vals = y_df.values.flatten() if y_df is not None else None
    indices = X_df.index.tolist()
    
    print(f"Generating features for {n_samples} samples...")
    
    for i in range(n_samples):
        row = X_records[i]
        pass_id = indices[i]
        sender_id = int(row['sender_id'])
        s_x, s_y = row[f'x_{sender_id}'], row[f'y_{sender_id}']
        
        # Determine attack direction (Home 1-11 attacks +X, Away 12-22 attacks -X)
        attack_dir = 1 if sender_id <= 11 else -1
        
        true_receiver = int(y_vals[i]) if y_vals is not None else None
        
        # Pre-calculate Opponent Positions for this pass
        opponents = range(12, 23) if sender_id <= 11 else range(1, 12)
        opp_coords = [(row[f'x_{oid}'], row[f'y_{oid}']) for oid in opponents]

        for candidate_id in range(1, 23):
            # Optional: Skip sender (you can't pass to yourself usually)
            # But the dataset format allows it (1-22). We'll keep them but they usually have dist=0.
            if candidate_id == sender_id:
                pass 

            c_x, c_y = row[f'x_{candidate_id}'], row[f'y_{candidate_id}']
            
            # --- Basic Features ---
            dist = get_distance(s_x, s_y, c_x, c_y)
            angle = get_angle(s_x, s_y, c_x, c_y)
            team = is_same_team(sender_id, candidate_id)
            
            # --- Advanced Features ---
            # Forward Progress: Higher is better
            forward_prog = (c_x - s_x) * attack_dir
            
            # Pressure: Distance to nearest opponent
            min_dist_opp = 100000
            for o_x, o_y in opp_coords:
                d_opp = get_distance(c_x, c_y, o_x, o_y)
                if d_opp < min_dist_opp:
                    min_dist_opp = d_opp
            
            entry = {
                'pass_id': pass_id,
                'candidate_id': candidate_id,
                'sender_id': sender_id,
                # Features
                'feat_dist': dist,
                'feat_angle': angle,
                'feat_same_team': team,
                'feat_forward_prog': forward_prog,
                'feat_pressure': min_dist_opp,
            }
            
            # Add target if training
            if true_receiver is not None:
                entry['target'] = 1 if candidate_id == true_receiver else 0
                
            data.append(entry)

    df = pd.DataFrame(data)

    # --- 3. Ranking Features (Critical for Accuracy) ---
    # We rank candidates per pass_id
    
    # Rank by Distance (Ascending: 1 = Closest)
    df['rank_dist'] = df.groupby('pass_id')['feat_dist'].rank(method='min')
    
    # Rank by Forward Progress (Descending: 1 = Most Forward)
    df['rank_forward'] = df.groupby('pass_id')['feat_forward_prog'].rank(method='min', ascending=False)
    
    # Rank by Pressure (Descending: 1 = Most Open / furthest from opponent)
    df['rank_pressure'] = df.groupby('pass_id')['feat_pressure'].rank(method='min', ascending=False)
    
    return df