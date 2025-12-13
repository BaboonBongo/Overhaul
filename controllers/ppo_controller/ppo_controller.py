from controller import Supervisor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import os

# --- PPO Implementation ---

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), # Bigger brain
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Output -1 to 1
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # Increased entropy weight to encourage exploration: 0.05
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.05 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- Webots Environment ---

class RobotEnvironment:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        self.robot_node = self.supervisor.getSelf()
        self.box_node = self.supervisor.getFromDef("box")
        
        self.motors = []
        self.sensors = []
        for i in range(1, 4):
            motor = self.supervisor.getDevice(f"joint{i}")
            # Position control but we will update positions incrementally to simulate velocity
            motor.setPosition(float('inf')) 
            motor.setVelocity(0.0)
            self.motors.append(motor)
            
            sensor = self.supervisor.getDevice(f"sensor{i}")
            sensor.enable(self.timestep)
            self.sensors.append(sensor)
            
        # GPS
        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(self.timestep)
            
        self.state_dim = 15 # 3sin + 3cos + 3box + 3ee + 3tgt
        self.action_dim = 3
        
        self.start_box_pos = [0.5, 0.025, 0.5]
        self.target_pos = [-0.5, 0.0, -0.5]
        
    def reset(self):
        self.robot_node.setVelocity([0,0,0,0,0,0])
        
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        if self.box_node:
            self.box_node.getField("translation").setSFVec3f(self.start_box_pos)
            self.box_node.getField("rotation").setSFRotation([0, 1, 0, 0])
            self.box_node.setVelocity([0,0,0,0,0,0])
            self.box_node.resetPhysics()
            
        for _ in range(10):
            self.supervisor.step(self.timestep)
            
        return self.get_state()
        
    def get_state(self):
        # 1. Joint States
        joint_vals = [s.getValue() for s in self.sensors]
        sin_q = np.sin(joint_vals)
        cos_q = np.cos(joint_vals)
        
        # 2. Positions
        box_pos = self.box_node.getPosition()
        ee_pos = self.gps.getValues()
        
        # 3. Target Vector
        rel_target = [t - b for t, b in zip(self.target_pos, box_pos)]
        
        state = np.concatenate([sin_q, cos_q, box_pos, ee_pos, rel_target])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        max_speed = 3.0 # Increase speed slightly
        
        # Apply Actions
        for i, motor in enumerate(self.motors):
            vel = np.clip(action[i], -1, 1) * max_speed
            
            # Simple Velocity Control
            if vel >= 0:
                motor.setPosition(float('inf'))
                motor.setVelocity(vel)
            else:
                motor.setPosition(-float('inf'))
                motor.setVelocity(abs(vel))
                
        self.supervisor.step(self.timestep)
        
        # --- Simple Dense Reward (No Ambiguity) ---
        box_pos = self.box_node.getPosition()
        ee_pos = self.gps.getValues()
        
        # Ranges: Distance ~ 0.5m - 1.0m
        dist_ee_box = np.linalg.norm([e - b for e, b in zip(ee_pos, box_pos)])
        dist_box_tgt = np.linalg.norm([b - t for b, t in zip(box_pos, self.target_pos)])
        
        # Reward: Minimize these distances.
        # Max penalty per step ~ -2.0. Min penalty ~ 0.
        reward = -(dist_ee_box * 2.0) - (dist_box_tgt * 4.0)
        
        done = False
        
        # Success Bonus
        if dist_box_tgt < 0.2:
            reward += 200.0
            print("Target Reached! $$$")
            done = True
            
        # Box Fell
        if box_pos[1] < -0.1: 
            reward -= 50.0
            done = True
            
        return self.get_state(), reward, done, {}

# --- Main Training Loop ---

def main():
    print("Starting PPO Training (High Exploration Mode)...")
    env = RobotEnvironment()
    # env.supervisor.simulationSetMode(env.supervisor.SIMULATION_MODE_FAST) # Wait for confirmation
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    max_episodes = 2000
    max_timesteps = 500
    update_timestep = 2000
    
    lr = 0.001       # Aggressive Learning Rate
    gamma = 0.99
    K_epochs = 20
    eps_clip = 0.2
    
    action_std = 1.0 # High Exploration Noise Start
    decay_rate = 0.05
    min_action_std = 0.1
    
    ppo_agent = PPO(state_dim, action_dim, lr, lr, gamma, K_epochs, eps_clip, action_std)
    
    time_step = 0
    episode_rewards = []
    
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(max_timesteps):
            time_step += 1
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            current_ep_reward += reward
            
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
            if done:
                break
        
        # Decay Action Std
        if i_episode % 25 == 0:
            action_std = action_std - decay_rate
            action_std = round(max(action_std, min_action_std), 2)
            ppo_agent.set_action_std(action_std)
                
        episode_rewards.append(current_ep_reward)
        
        if i_episode % 10 == 0:
            avg_rew = np.mean(episode_rewards[-10:])
            print(f"Episode {i_episode} | Avg Reward: {avg_rew:.2f} | Std: {action_std}")
            
        if i_episode % 100 == 0:
            ppo_agent.save(f"ppo_model_{i_episode}.pth")

    env.supervisor.simulationSetMode(env.supervisor.SIMULATION_MODE_REAL_TIME)
    
    plt.plot(episode_rewards)
    plt.savefig("final_reward.png")
    print("Training finished.")

if __name__ == '__main__':
    main()
