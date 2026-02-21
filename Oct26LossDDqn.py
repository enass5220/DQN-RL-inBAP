import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Define the environment
class PortEnv(gym.Env):
    def __init__(self, dataset):
        super().__init__()
        self.source_dataset = dataset.copy(deep=True)
        
        # This is the working copy for the current episode. It will be refreshed from source.
        self.dataset = self.source_dataset.copy(deep=True)
        
        self.current_step = 0
        self.berths = {
            'a': 200, 
            'b': 184.22, 'c': 184.22, 'd': 184.22, 'e': 184.22, 'f': 184.22,
            'g': 184.5, 'h': 184.5, 'i': 184.5, 'j': 184.5,
            'ja': 200, 'k': 200,'ka': 185.35, 'l': 185.35, 'n': 185.35,
            'o': 240,
            'p': 200.45, 'q': 200.45, 'r': 200.45, 's': 200.45,
            't': 1422.45
        }
        self.max_steps = 1000 #428
        self.all_done = False
        self.max_waiting_time = 24 * 7 *2  # two weeks
        # Cargo type constraints
        self.cargo_constraints = {
             'E': ['s', 'r'],
            'DA': ['e'],
            'D': ['d', 'b', 'c', 'h', 'g', 'i', 'ja'],
            'CB': ['b','g','k','ka','m','n','p'],
            'C': ['p', 'a', 'b', 'c', 'd', 'ka', 'g', 'h', 'i', 'j', 'k', 'l'],
            'B': ['k', 'ka', 'm', 'n','l', 'b', 'c', 'd', 'g', 'h', 'i', 'j', 'ja', 'p'],
            'F': ['t'],
            'A': ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'ja', 'ka', 'q']
            # TODO change cargo types according to real data
            # for exxample, remove car berths from cargo type A
        }

        # Adjacent berths
        self.adjacent_berths = {
            'b': ['c'], 'c': ['b', 'd'], 'd': ['c', 'e'], 'e': ['d', 'f'], 'f': ['e'],
            'g': ['h'], 'h': ['g', 'i'], 'i': ['h', 'j'], 'j': ['i'],
            'ja': ['k'], 'k': ['ja'],
            'ka': ['l'], 'l': ['ka', 'n'], 'n': ['l'],
            'p': ['q'], 'q': ['p', 'r'], 'r': ['q', 's'], 's': ['r']
        }

        # Calculate num_features based on the state representation
        num_vessel_features = 3  # length_norm, waiting_time_norm, service_hours_norm
        num_berth_features = len(self.berths) * 2  # Berth available length and occupancy status
        num_time_features = 1  # Current time (normalized)
        num_cargo_features = len(self.cargo_constraints)  # One-hot encoded cargo type

        self.action_space = spaces.Discrete(len(self.berths))  # Assign to one of the berths
        observation_shape = (num_berth_features + num_time_features + num_vessel_features + num_cargo_features,)
        self.observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.float32)

        # Initialize state
        self.state = self._get_initial_state()

        # Sort the dataset by the original arrival time.
        self.sorted_dataset = self.dataset.sort_values(by='arrival').reset_index(drop=True)
        self.vessel_queue = self.sorted_dataset.to_dict(orient='records')
        
        # Define a scaling factor to convert real-world time to simulation time. This is crucial!
        self.time_scale_factor = 3600  # This equals 1 hour.  1 means 1 simulation step = 1 second.
        # You can adjust this factor depending on how fast you want your simulation to run
        # Larger factor will compress the simulation, smaller factor will expand it.

        # Pre-calculate scaled arrival times and service hours and store them in the vessel data.
        self._scale_time_data()
        self.assigned_vessels_log = [] # Create a log to store data for all assigned vessels

        # Waiting queue for vessels that cannot be immediately assigned.
        self.waiting_queue = []
        
        self.final_metrics = {}

    def _scale_time_data(self):
        """Scales the arrival and service times in the dataset to the simulation time scale."""
        first_arrival_time = self.sorted_dataset['arrival_ts'].min() #get minimum ts to set as zero
        # CRITICAL CHANGE: Apply scaling to the DataFrame directly
        self.dataset['arrival_ts_scaled'] = (self.dataset['arrival_ts'] - first_arrival_time) / self.time_scale_factor
        self.dataset['service_hours_scaled'] = self.dataset['service_hours']
        self.dataset['service_hours_norm_scaled'] = self.dataset['service_hours_norm'] / self.time_scale_factor

    def reset(self, seed=None, options=None):
        # Always reset from the original source data
        self.dataset = self.source_dataset.copy(deep=True)
        self.current_step = 0
        self.state = self._get_initial_state()
        
        self._scale_time_data() # Scale the fresh data copy
        
        # Now sort the freshly scaled data
        self.dataset.sort_values(by='arrival', inplace=True)
        self.vessel_queue = self.dataset.to_dict(orient='records')
        
        self.waiting_queue = []
        self.assigned_vessels_log = []
        self.final_metrics = {}
        
        observation = self._prepare_observation(vessel=None)
        return observation, {}


    def _prepare_observation(self, vessel):
        if vessel is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Berth occupancy (available length and occupancy status)
        berth_features = []
        for berth, berth_info in self.state['berth_occupancy'].items():
            if berth_info['occupied']:
                # If berth is occupied, include remaining available length and occupancy status (1 for occupied)
                berth_features.extend([berth_info['available_length'] / self.berths[berth], 1])  # Normalize by total length
            else:
                # If berth is not occupied, include total length as available and occupancy status (0 for unoccupied)
                berth_features.extend([1, 0])  # 1 means fully available, 0 means unoccupied

        berth_occupancy = np.array(berth_features, dtype = np.float32)

        # Current time (normalized)
        current_time = np.array([self.state['time'] / self.max_steps], dtype = np.float32)

        # Vessel features (length, waiting_time, service_hours)
        waiting_time = (self.state['time'] - vessel['arrival_ts_scaled']) if self.state['time'] >= vessel['arrival_ts_scaled'] else 0
        waiting_time = min(waiting_time, self.max_waiting_time)
        waiting_time_norm = waiting_time / self.max_waiting_time

        vessel_features = np.array([vessel['length'] / max(self.berths.values()), waiting_time_norm, vessel['service_hours'] / self.max_steps], dtype = np.float32)

        # One-hot encoded cargo type
        cargo_type = vessel['cargo_type']
        cargo_one_hot = np.zeros(len(self.cargo_constraints), dtype = np.float32)  # Initialize with zeros
        cargo_index = list(self.cargo_constraints.keys()).index(cargo_type)  # Get index of cargo type
        cargo_one_hot[cargo_index] = 1  # Set the corresponding index to 1

        # Combine all features into a single observation vector
        observation = np.concatenate([berth_occupancy, current_time, vessel_features, cargo_one_hot]).astype(np.float32)
        return observation

    def _calculate_waiting_time(self, vessel):
        """
        Calculate waiting time for the vessel in hours based on simulation time.
        """
        if 'assignment_time' in vessel and 'waiting_start_time' in vessel:  # check if vessel has been assigned a berth and has a waiting start time
            waiting_time = vessel['assignment_time'] - vessel['waiting_start_time']
            return waiting_time
        return 0  # Vessel hasn't been assigned

    def step(self, action):
        reward = 0

        vessel = self._get_next_arrived_vessel()
        if not vessel:
            # No new vessel, no action to take, no reward or penalty
            pass
        else:
            if 'arrival_time' not in vessel: vessel['arrival_time'] = self.state['time']
            berth_assigned = list(self.berths.keys())[action]
            if not self._is_action_valid(vessel, berth_assigned):
                self.waiting_queue.append(vessel)
                # Give a small penalty for invalid moves to discourage them
                reward = -1.0 
            else:
                self._update_state(vessel, berth_assigned)
                # No immediate reward for a valid move. The reward comes at the end.
        
        self.state['time'] += 1
        self.current_step += 1
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps
        
        # --- NEW REWARD LOGIC: Grant reward only at the end ---
        if terminated or truncated:
            # When the episode is over, calculate the final reward
            # The goal is to minimize total waiting time. So, the reward is the NEGATIVE of total waiting time.
            # We normalize it to keep the values from getting too large.
            log_df = pd.DataFrame(self.assigned_vessels_log)
            if not log_df.empty:
                log_df['waiting_time'] = log_df['assignment_time'] - log_df['arrival_time']
                total_waiting_time = log_df['waiting_time'].sum()
                # A simple normalization factor
                reward = -total_waiting_time / len(self.source_dataset) 
            else:
                reward = 0

        self._free_departed_berths()
        self._assign_waiting_vessels()
        
        next_vessel_for_obs = self.waiting_queue[0] if self.waiting_queue else (self.vessel_queue[0] if self.vessel_queue else None)
        observation = self._prepare_observation(next_vessel_for_obs)

        return observation, reward, terminated, truncated, {}

    def _is_done(self):
        done = not self.vessel_queue and not self.waiting_queue 
        if done:
            self.all_done = True
        return done

    def _is_action_valid(self, vessel, berth_assigned):
        cargo_type = vessel['cargo_type']
        berth = self.berths[berth_assigned]
        allowed_berths = self.cargo_constraints.get(cargo_type, list(self.berths.keys()))

        if berth_assigned not in allowed_berths:
            return False

        vessel_length = vessel['length']
        berth_length = self.state['berth_occupancy'][berth_assigned]['available_length']

        if vessel_length <= berth_length:
            return True
        elif berth_length <=0.9:
            return False
        else:
            adjacent_berths = self.adjacent_berths.get(berth_assigned, [])
            for adj_berth in adjacent_berths:
                if adj_berth in allowed_berths and self.berths[adj_berth] >= (vessel_length - berth_length):
                    if (self.state['berth_occupancy'][adj_berth]['available_length'] + self.state['berth_occupancy'][berth_assigned]['available_length'])>= vessel['length']:
                        return True
            return False

    def _calculate_berth_utilization(self, berth_assigned):
        current_time = self.state['time']
        berth_occupancy = self.state['berth_occupancy']

        if berth_occupancy[berth_assigned]['occupied']:
            occupancy_time = berth_occupancy[berth_assigned]['occupied_vessel']['service_hours_scaled']
            vessel_length = berth_occupancy[berth_assigned]['occupied_vessel']['length']
            berth_length = self.berths[berth_assigned]
            utilization = (vessel_length / berth_length) * (occupancy_time / current_time) if current_time > 0 else 0.0
            return min(utilization, 1.0)
        else:
            return 0.0
        
    def _calculate_global_berth_utilization(self):
        total_berth_length = sum(self.berths.values())
        total_occupied_space_time = 0

        for berth, berth_info in self.state['berth_occupancy'].items():
            if berth_info['occupied']:
                vessel = berth_info['occupied_vessel']
                occupied_length = vessel['length']
                service_time = vessel['service_hours_scaled']
                total_occupied_space_time += occupied_length * service_time

        global_utilization = total_occupied_space_time / (total_berth_length * self.state['time']) if self.state['time'] > 0 else 0.0
        return min(global_utilization, 1.0)
    
    def _get_next_arrived_vessel(self):
        while self.vessel_queue:
            next_vessel = self.vessel_queue[0]
            arrival_time = next_vessel['arrival_ts_scaled']
            if self.state['time'] >= arrival_time:
                return self.vessel_queue.pop(0)
            else:
                self.state['time'] = arrival_time
                break
        return None

    def _update_state(self, vessel, berth_assigned):
        berth_length = self.berths[berth_assigned]
        vessel['assignment_time'] = self.state['time']
        self.state['berth_occupancy'][berth_assigned]['occupied'] = True
        self.state['berth_occupancy'][berth_assigned]['mooring_ts'] = self.state['time']
        vessel_length = vessel['length']
        vessel['departure_time'] = vessel['assignment_time'] + vessel['service_hours']
        if vessel_length <= self.state['berth_occupancy'][berth_assigned]['available_length']:
            occupied_length = vessel['length']
            self.state['berth_occupancy'][berth_assigned]['occupied_length'] = occupied_length
            self.state['berth_occupancy'][berth_assigned]['available_length'] -= vessel['length']
            self.state['berth_occupancy'][berth_assigned]['occupied_vessel'] = vessel
        else:
            adjacent_berths = self.adjacent_berths.get(berth_assigned, [])
            for adj_berth in adjacent_berths:
                if adj_berth in self.cargo_constraints.get(vessel['cargo_type'], list(self.berths.keys())):
                    adj_available_length = self.state['berth_occupancy'][adj_berth]['available_length']
                    current_available_length = self.state['berth_occupancy'][berth_assigned]['available_length']
                    available_space = current_available_length + adj_available_length
                    if available_space >= vessel['length']:
                        adj_departure_time = vessel['assignment_time'] + vessel['service_hours']
                        current_occupied_length = current_available_length
                        self.state['berth_occupancy'][berth_assigned]['occupied_vessel'] = vessel
                        adj_occupied_length = vessel['length'] - current_available_length
                        self.state['berth_occupancy'][adj_berth]['occupied_length'] = adj_occupied_length
                        self.state['berth_occupancy'][adj_berth]['available_length'] -= adj_occupied_length
                        self.state['berth_occupancy'][adj_berth]['occupied'] = True
                        self.state['berth_occupancy'][adj_berth]['occupied_vessel'] = vessel
                        self.state['berth_occupancy'][berth_assigned]['occupied_length'] = current_occupied_length
                        self.state['berth_occupancy'][berth_assigned]['available_length'] = 0
                        self.state['berth_occupancy'][berth_assigned]['departure_time'] = vessel['assignment_time'] + vessel['service_hours']
        self.current_step += 1
        self.assigned_vessels_log.append(vessel)
        return self.state

    def _get_initial_state(self):
        self.max_steps = 1000
        self.all_done=False
        state = {
            'time': 0,
            'berth_occupancy': {
                berth: {
                    'occupied': False,
                    'available_length': self.berths[berth],
                    'occupied_length': 0
                }
                for berth in self.berths
            },
            'vessel_queue': []
        }
        return state
    
    def _free_departed_berths(self):
        for berth, berth_info in self.state['berth_occupancy'].items():
            if berth_info['occupied'] and berth_info['occupied_vessel'] is not None:
                if 'departure_time' not in berth_info['occupied_vessel']:
                    berth_info['occupied_vessel']['departure_time'] = berth_info['occupied_vessel']['assignment_time'] + berth_info['occupied_vessel']['service_hours']
                if berth_info['occupied_vessel']['departure_time'] <= self.state['time']:
                    vessel = berth_info['occupied_vessel']
                    adjacent_berths = self.adjacent_berths.get(berth, [])
                    for adj_berth in adjacent_berths:
                        if self.state['berth_occupancy'][adj_berth]['occupied'] and self.state['berth_occupancy'][adj_berth]['occupied_vessel'] is not None and self.state['berth_occupancy'][adj_berth]['occupied_vessel']['imo'] == vessel['imo']:
                            self.state['berth_occupancy'][adj_berth] = {
                                'occupied': False,
                                'available_length':  self.berths[adj_berth],
                                'occupied_length': 0,
                                'occupied_vessel': None,
                                'mooring_ts': None,
                                'departure_time': None
                            }
                    self.state['berth_occupancy'][berth] = {
                        'occupied': False,
                        'available_length': self.berths[berth],
                        'occupied_length': 0,
                        'occupied_vessel': None,
                        'mooring_ts': None,
                        'departure_time': None
                    }

    def _assign_waiting_vessels(self):
        i = 0
        while i < len(self.waiting_queue):
            # assigned=False
            vessel = self.waiting_queue[i]
            for j, berth_name in enumerate(self.berths.keys()):
                if (self._is_action_valid(vessel, berth_name)):
                    self._update_state(vessel, berth_name)
                    self.waiting_queue.pop(i)
                    return
            i=i+1
            if len(self.waiting_queue) == i:
                break
    
    def calculate_and_log_metrics(self):
        if not self.assigned_vessels_log:
            print("No vessels were assigned. Cannot calculate metrics.")
            return
        total_berth_length_time = sum(self.berths.values()) * self.state['time']
        total_occupied_space_time = sum(v['length'] * v['service_hours'] for v in self.assigned_vessels_log)
        berth_utilization_rate = total_occupied_space_time / total_berth_length_time if total_berth_length_time > 0 else 0
        self.final_metrics['Berth Utilization Rate'] = berth_utilization_rate
        log_df = pd.DataFrame(self.assigned_vessels_log)
        log_df['waiting_time'] = log_df['assignment_time'] - log_df['arrival_time']
        waiting_times = log_df['waiting_time']
        log_df[['imo','arrival_time', 'assignment_time', 'waiting_time']].to_csv(
        "dqn_vessel_log.csv", index=False)
        positive_waiting_times = waiting_times[waiting_times > 0]
        if not positive_waiting_times.empty:
            avg_wait, sem_wait, max_wait, min_wait, median_wait = positive_waiting_times.mean(), positive_waiting_times.sem(), positive_waiting_times.max(), positive_waiting_times.min(), positive_waiting_times.median()
        else:
            avg_wait, sem_wait, max_wait, min_wait, median_wait = 0, 0, 0, 0, 0
        self.final_metrics['Overall Waiting Time'] = {
            'Total Vessel Waiting Time': waiting_times.sum(),
            'Average Vessel Waiting Time': avg_wait,
            'Median Vessel Waiting Time': median_wait, # NEW
            'Standard Error of Vessel Waiting Time': sem_wait,
            'Maximum Vessel Waiting Time': max_wait,
            'Shortest Vessel Waiting Time': min_wait
        }
        self.final_metrics['Waiting Time by Cargo Type'] = {}
        for cargo_type, times in log_df.groupby('cargo_type')['waiting_time']:
            positive_times = times[times > 0]
            if not positive_times.empty:
                avg, sem, max_t, min_t, med = positive_times.mean(), positive_times.sem(), positive_times.max(), positive_times.min(), positive_times.median()
            else:
                avg, sem, max_t, min_t, med = 0, 0, 0, 0, 0
            self.final_metrics['Waiting Time by Cargo Type'][cargo_type] = {
                'Total Vessel Waiting Time': times.sum(),
                'Average Vessel Waiting Time': avg,
                'Median Vessel Waiting Time': med, # NEW
                'Standard Error of Vessel Waiting Time': sem,
                'Maximum Vessel Waiting Time': max_t,
                'Shortest Vessel Waiting Time': min_t
            }

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        self.features_extractor = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        extracted_features = self.features_extractor(observations)
        return extracted_features


class TrainingProgressCallback(BaseCallback):
    """
    A custom callback to log and plot training progress.
    This version robustly fetches the loss from the model's logger,
    which is the correct way for off-policy algorithms like DQN.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.reward_timesteps, self.rewards = [], []
        self.loss_timesteps, self.losses = [], []
        self.ep_rew_buffer = []
        # NEW: Lists for episode length
        self.ep_len_timesteps, self.ep_lengths = [], []
        self.ep_len_buffer = []

    def _on_step(self) -> bool:
        # The Monitor wrapper adds 'episode' to the info dict when an episode is done
        if "dones" in self.locals:
            for i, done in enumerate(self.locals['dones']):
                if done and 'episode' in self.locals['infos'][i]:
                    info = self.locals['infos'][i]['episode']
                    self.ep_rew_buffer.append(info['r'])
                    # NEW: Log the episode length ('l' is the key for length)
                    self.ep_len_buffer.append(info['l'])
                    if len(self.ep_rew_buffer) > 10: self.ep_rew_buffer.pop(0)
                    if len(self.ep_len_buffer) > 10: self.ep_len_buffer.pop(0)
        
        if self.num_timesteps % self.check_freq == 0:
            if self.ep_rew_buffer:
                mean_reward = np.mean(self.ep_rew_buffer)
                self.rewards.append(mean_reward)
                self.reward_timesteps.append(self.num_timesteps)
            # NEW: Calculate and store mean episode length
            if self.ep_len_buffer:
                mean_length = np.mean(self.ep_len_buffer)
                self.ep_lengths.append(mean_length)
                self.ep_len_timesteps.append(self.num_timesteps)

            loss = self.model.logger.name_to_value.get('train/loss', np.nan)
            if not np.isnan(loss):
                self.losses.append(loss)
                self.loss_timesteps.append(self.num_timesteps)
        return True
    
def log_metrics_to_files(metrics, total_reward, model_label, timestamp):
    log_dir = "./rl_logs/"
    os.makedirs(log_dir, exist_ok=True)
    txt_filepath = os.path.join(log_dir, f"results_{model_label}_{timestamp}.txt")
    with open(txt_filepath, 'w') as f:
        f.write(f"--- Performance Metrics for {model_label} ---\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Total Reward: {total_reward:.2f}\n")
        if 'Berth Utilization Rate' in metrics: f.write(f"Berth Utilization Rate: {metrics['Berth Utilization Rate']:.2%}\n")
        if 'Overall Waiting Time' in metrics:
            f.write("\n--- Overall Waiting Time Metrics ---\n")
            for key, value in metrics['Overall Waiting Time'].items(): f.write(f"  {key}: {value:.2f}\n")
        if 'Waiting Time by Cargo Type' in metrics:
            f.write("\n--- Waiting Time Metrics by Cargo Type ---\n")
            for cargo_type, data in metrics['Waiting Time by Cargo Type'].items():
                f.write(f"\n  Cargo Type: {cargo_type}\n")
                for key, value in data.items(): f.write(f"    {key}: {value:.2f}" if pd.notna(value) else f"    {key}: N/A\n")
    print(f"Human-readable results saved to {txt_filepath}")
    records = []
    records.append({'Model': model_label, 'Timestamp': timestamp, 'Category': 'Overall', 'Metric': 'Total Reward', 'Value': total_reward})
    if 'Berth Utilization Rate' in metrics: records.append({'Model': model_label, 'Timestamp': timestamp, 'Category': 'Overall', 'Metric': 'Berth Utilization Rate', 'Value': metrics['Berth Utilization Rate']})
    if 'Overall Waiting Time' in metrics:
        for key, value in metrics['Overall Waiting Time'].items(): records.append({'Model': model_label, 'Timestamp': timestamp, 'Category': 'Overall', 'Metric': key, 'Value': value})
    if 'Waiting Time by Cargo Type' in metrics:
        for cargo_type, data in metrics['Waiting Time by Cargo Type'].items():
            for key, value in data.items(): records.append({'Model': model_label, 'Timestamp': timestamp, 'Category': cargo_type, 'Metric': key, 'Value': value})
    csv_filepath = os.path.join(log_dir, f"results_summary{timestamp}.csv") # Changed to a single summary file
    results_df = pd.DataFrame(records)
    if os.path.exists(csv_filepath):
        results_df.to_csv(csv_filepath, mode='a', header=False, index=False)
    else:
        results_df.to_csv(csv_filepath, mode='w', header=True, index=False)
    print(f"Structured results appended to {csv_filepath}")    
     
# Load dataset
df = pd.read_csv("y2024.csv") #"2024.csv"

# Convert datetime columns
df["arrival"] = pd.to_datetime(df["arrival"], format="%d-%m-%Y %H:%M")

# Convert datetime to timestamps
df["arrival_ts"] = df["arrival"].astype(int) / 10 ** 9
df["cargo_type"] = df["cargo_cat"].astype(str)
# One-hot encode cargo_cat
encoder = OneHotEncoder()
cargo_encoded = encoder.fit_transform(df[["cargo_type"]])
cargo_encoded_df = pd.DataFrame(cargo_encoded.toarray(), columns=encoder.get_feature_names_out(["cargo_type"]))

# Normalize numerical features
scaler = MinMaxScaler()
numerical_cols = ["length", "service_hours"] + (["waiting_time"] if "waiting_time" in df.columns else [])
numerical_features = df[["length", "service_hours", "waiting_time"]]
normalized_features = scaler.fit_transform(numerical_features)
normalized_features_df = pd.DataFrame(normalized_features, columns=["length_norm", "service_hours_norm", "waiting_time_norm"])

# Combine all features
processed_df = pd.concat([df, cargo_encoded_df, normalized_features_df], axis=1)
# print
print(f"DataFrame loaded with {len(processed_df)} vessels.")
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style
sns.set_theme(style="whitegrid")

# Define the order of cargo types to make the plot look professional
cargo_order = ['A', 'B', 'C', 'CB', 'D', 'DA', 'E', 'F']

plt.figure(figsize=(14, 8))

# Create the Box Plot
# 'patch_artist' allows us to fill the boxes with color
ax = sns.boxplot(
    data=processed_df, 
    x='cargo_type', 
    y='service_hours', 
    order=cargo_order,
    palette='Set2',
    showfliers=True  # Ensure outliers are visible as these are your "hidden killers"
)

# Overlay a stripplot to show individual vessel "dots" (optional but looks great)
sns.stripplot(
    data=processed_df, 
    x='cargo_type', 
    y='service_hours', 
    order=cargo_order,
    color='black', 
    size=3, 
    alpha=0.3
)

# Customizing the labels for the thesis
plt.title('Vessel Service Duration by Cargo Category (2022–2024)', fontsize=16, fontweight='bold')
plt.xlabel('Cargo Category Code', fontsize=13)
plt.ylabel('Service Duration (Hours)', fontsize=13)

# Adding gridlines for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save the figure for your Chapter 3
plt.savefig("./rl_logs/cargo_vs_service_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
#input()
# Create environment
env = PortEnv(processed_df)

# Policy kwargs for the custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(),
    net_arch=[256, 256] #[128, 128] 
)
# ==============================================================================
#      1. TRAINING PHASE
# ==============================================================================

# Create directory for logs
LOG_DIR = "./rl_logs/"
os.makedirs(LOG_DIR, exist_ok=True)
callback = TrainingProgressCallback(check_freq=1000, log_dir=LOG_DIR)

# --- Train dqn RL agent ---
# Note the changes in hyperparameters for DQN
model = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.00001,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=64,
        gamma=0.95,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10000,
    exploration_fraction=0.5,
    exploration_final_eps=0.05,
    policy_kwargs=policy_kwargs,
    # Double Q-learning is True by default 
)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = f"berth_allocation_agent_dqn_{timestamp}"
# best one so far:
## berth_allocation_agent_dqn_2025-10-26_20-25-35
# check loss and reward logs of same name in rl_logs folder

model.learn(total_timesteps=300000, callback=callback)
model.save(model_path)
print("DONE TRAINING")

# ==========================================================
#         PLOTTING CODE - RUNS AFTER TRAINING
# ==========================================================

# --- Plot 1: Mean Reward vs. Timesteps ---
plt.figure(figsize=(12, 6))
plt.plot(callback.reward_timesteps, callback.rewards, color='blue')
plt.title('Training Progress: Mean Episode Reward')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.savefig(os.path.join(LOG_DIR, f'reward_progress_graph_dqn_{timestamp}.png'), dpi=300)
#plt.savefig(os.path.join(LOG_DIR, f'reward_progress_graph_{timestamp}.pdf'))
print(f"\nReward graph saved to {LOG_DIR}")
plt.show()


# --- Plot 2: Smoothed Training Loss vs. Timesteps ---
# This plot shows if the agent's value predictions are converging.
# A decreasing and stabilizing curve indicates that the learning process is stable.
plt.figure(figsize=(12, 6))
if callback.losses:
    loss_series = pd.Series(callback.losses)
    # We apply a rolling average to smooth the curve and see the underlying trend.
    windows = loss_series.rolling(window=10, min_periods=1)
    smoothed_losses = windows.mean()
    smoothed_timesteps = pd.Series(callback.loss_timesteps).rolling(window=10, min_periods=1).mean()
    
    plt.plot(smoothed_timesteps, smoothed_losses, color='red')
    plt.title('Training Progress: Smoothed Training Loss')
    plt.xlabel('Timesteps')
    plt.ylabel('Smoothed Loss')
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, f'loss_progress_graph_dqn_{timestamp}.png'))
    plt.show()
else:
    print("No loss data was recorded by the callback.")

# NEW: Episode Length Plot
plt.figure(figsize=(10, 6)) # Create a new figure
if callback.ep_lengths:
    plt.plot(callback.ep_len_timesteps, callback.ep_lengths, color='green')
    plt.title('Training Progress: Mean Episode Length')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Steps per Episode (Nodes Visited)')
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f'training_summary_graphs_dqn_{timestamp}.png'))
plt.show()
# ==========================================================
#         PLOTTING CODE - END
# ==========================================================


# Load the model you just trained
 
model = DQN.load(model_path)

# ==============================================================================
#      2. TESTING PHASE
# ==============================================================================
print("\n--- TESTING PHASE ON NEW DATASET ---")
try:
    test_df = pd.read_csv("2023.csv") 
    
    test_df["arrival"] = pd.to_datetime(test_df["arrival"], format='mixed', dayfirst=True)
    test_df["arrival_ts"] = test_df["arrival"].astype(int) / 10**9
    test_df['cargo_type'] = test_df['cargo_cat'].astype(str).str.replace(r'\s+', '', regex=True).str.upper()
    if 'imo' not in test_df.columns: test_df['imo'] = range(len(test_df))
    test_df['imo'].fillna(pd.Series(range(len(test_df))), inplace=True)

    # Use the SAME scaler from training. Only TRANSFORM.
    test_normalized_features = scaler.transform(test_df[numerical_cols])
    test_normalized_features_df = pd.DataFrame(test_normalized_features, columns=[f"{c}_norm" for c in numerical_cols])
    processed_test_df = pd.concat([test_df.reset_index(drop=True), test_normalized_features_df], axis=1)
    print(f"Testing DataFrame loaded with {len(processed_test_df)} vessels.")

    test_env = PortEnv(processed_test_df)
    model = DQN.load(model_path)
    obs, info = test_env.reset()
    total_reward = 0
    for _ in range(200000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    test_env.calculate_and_log_metrics()
    test_metrics = test_env.final_metrics

    # --- Print and LOG metrics for the TESTING Data ---
    print("\n--- Final Performance Metrics on TESTING DATASET ---")
    print(f"Total Reward: {total_reward:.2f}")
    if 'Berth Utilization Rate' in test_metrics: print(f"Berth Utilization Rate: {test_metrics['Berth Utilization Rate']:.2%}")
    if 'Overall Waiting Time' in test_metrics:
        print("\n--- Overall Waiting Time Metrics ---")
        for key, value in test_metrics['Overall Waiting Time'].items(): print(f"  {key}: {value:.2f}")
    
    # *** THIS IS THE NEW PART ***
    log_metrics_to_files(test_metrics, total_reward, "DQN_Test_2023_Data", timestamp)

except FileNotFoundError:
    print("!!! TESTING FILE NOT FOUND. Please update the path. Skipping testing phase. !!!")