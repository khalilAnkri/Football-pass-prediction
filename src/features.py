import pandas as pd
import numpy as np

# --- 1. Helper Functions ---

def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def is_same_team(sender_id, player_id):
    if sender_id <= 11:
        return 1 if player_id <= 11 else 0
    else:
        return 1 if player_id > 11 else 0

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """
    Calculates the minimum distance from point P(px,py) to the line segment (x1,y1)-(x2,y2).
    Vectorized for single point P vs single segment.
    """
    # Vector AB (Segment)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line (parameter t)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)

    # Clamp t to segment [0, 1]
    t = max(0, min(1, t))

    # Closest point on segment
    cx = x1 + t * dx
    cy = y1 + t * dy

    return np.sqrt((px - cx)**2 + (py - cy)**2)

# --- 2. Main Feature Computation ---

def compute_features(X_df, y_df=None):
    n_samples = X_df.shape[0]
    data = []
    
    # Pre-convert to list of dicts for speed
    X_records = X_df.to_dict('records')
    y_vals = y_df.values.flatten() if y_df is not None else None
    indices = X_df.index.tolist()
    
    print(f"Generating PRO features (Blockage, Congestion) for {n_samples} passes...")
    
    for i in range(n_samples):
        row = X_records[i]
        pass_id = indices[i]
        s_id = int(row['sender_id'])
        s_x, s_y = row[f'x_{s_id}'], row[f'y_{s_id}']
        
        # Attack direction (+1 or -1)
        attack_dir = 1 if s_id <= 11 else -1
        
        true_receiver = int(y_vals[i]) if y_vals is not None else None
        
        # Get Opponent Coordinates
        opp_ids = range(12, 23) if s_id <= 11 else range(1, 12)
        opp_coords = [(row[f'x_{oid}'], row[f'y_{oid}']) for oid in opp_ids]

        for c_id in range(1, 23):
            # We keep it for consistency but expect low prob.
            
            c_x, c_y = row[f'x_{c_id}'], row[f'y_{c_id}']
            
            # --- Standard Features ---
            dist = get_distance(s_x, s_y, c_x, c_y)
            angle = get_angle(s_x, s_y, c_x, c_y)
            team = is_same_team(s_id, c_id)
            fwd_prog = (c_x - s_x) * attack_dir

            
            # 1. Congestion (Opponents within 3m / 300cm)
            congestion_count = 0
            # 2. Nearest Opponent Distance (Pressure)
            min_dist_opp = 10000.0
            # 3. Lane Blockage (Interception Risk)
            min_lane_dist = 10000.0
            # 4. Packing (Opponents bypassed)
            packing_count = 0

            for ox, oy in opp_coords:
                # Dist to candidate (Pressure)
                d_opp = get_distance(c_x, c_y, ox, oy)
                if d_opp < min_dist_opp:
                    min_dist_opp = d_opp
                
                # Congestion (3m radius)
                if d_opp < 300: 
                    congestion_count += 1
                
                # Lane Blockage
                # Dist from opponent to line segment (Sender -> Candidate)
                d_line = point_to_segment_dist(ox, oy, s_x, s_y, c_x, c_y)
                if d_line < min_lane_dist:
                    min_lane_dist = d_line
                
                # Packing (Simple definition: Opponent is between Sender and Receiver in X axis)
                # Check if opponent X is between Sender X and Candidate X
                if min(s_x, c_x) < ox < max(s_x, c_x):
                     packing_count += 1

            entry = {
                'pass_id': pass_id,
                'candidate_id': c_id,
                'sender_id': s_id,
                'feat_dist': dist,
                'feat_angle': angle,
                'feat_team': team,
                'feat_fwd': fwd_prog,
                'feat_pressure': min_dist_opp,
                'feat_congestion': congestion_count,
                'feat_blockage': min_lane_dist,
                'feat_packing': packing_count
            }
            
            if true_receiver is not None:
                entry['target'] = 1 if c_id == true_receiver else 0
                
            data.append(entry)

    df = pd.DataFrame(data)

    # --- 3. Ranking Features ---
    # Rank everything! The model loves relative comparisons.
    
    # Closest Teammate
    df['rank_dist'] = df.groupby('pass_id')['feat_dist'].rank(method='min')
    
    # Most Forward Teammate
    df['rank_fwd'] = df.groupby('pass_id')['feat_fwd'].rank(method='min', ascending=False)
    
    # Most Open Teammate (Highest distance to nearest opponent)
    df['rank_pressure'] = df.groupby('pass_id')['feat_pressure'].rank(method='min', ascending=False)
    
    # Least Blocked Teammate (Highest distance from lane to opponent)
    df['rank_blockage'] = df.groupby('pass_id')['feat_blockage'].rank(method='min', ascending=False)
    
    return df