import torch

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 num_obs,
                 num_critic_obs,
                 num_actions,
                 device='cpu'):

        self.device = device

        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, num_obs, device=self.device)
        if num_critic_obs is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, num_critic_obs,
                                                       device=self.device)
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(
            transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, 1))

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()

            advantage = self.rewards[step] + next_is_not_terminal * gamma * (
                        next_values + lam * advantage) - self.values[step]
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]

                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]

                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, \
                    target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
