"""Vanilla Policy Gradient (REINFORCE)."""
import collections
import copy

import torch
import torch.nn.functional as F
from dowel import tabular, logger
from garage import log_performance, TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos import BatchPolopt
from garage.torch.algos import (_Default, compute_advantages, filter_valids,
                                make_optimizer, pad_to_last)
from garage.torch.utils import flatten_batch


class VPG(BatchPolopt):
    """Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a
            tuple of type and dictionary, where dictionary contains arguments
            to initialize the optimizer e.g. `(torch.optim.Adam, {'lr' = 1e-3})`  # noqa: E501
        policy_lr (float): Learning rate for policy parameters.
        max_path_length (int): Maximum length of a single rollout.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        minibatch_size (int): Batch size for optimization.
        max_optimization_epochs (int): Maximum number of epochs for update.

    """

    def __init__(
            self,
            env_spec,
            policy,
            baseline,
            optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-2),
            max_path_length=_Default(500),
            num_train_per_epoch=_Default(1),
            discount=_Default(0.99),
            gae_lambda=_Default(1),
            center_adv=_Default(True),
            positive_adv=_Default(False),
            policy_ent_coeff=_Default(0.0),
            use_softplus_entropy=_Default(False),
            stop_entropy_gradient=_Default(False),
            entropy_method=_Default('no_entropy'),
            minibatch_size=_Default(None),
            max_optimization_epochs=_Default(1),
    ):
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._minibatch_size = minibatch_size
        self._max_optimization_epochs = max_optimization_epochs
        self._eps = 1e-8

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self._optimizer = make_optimizer(optimizer,
                                         policy,
                                         lr=policy_lr,
                                         eps=_Default(1e-5))

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=num_train_per_epoch)

        self._old_policy = copy.deepcopy(self.policy)

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, itr, paths):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """
        obs, actions, rewards, valids, baselines = self.process_samples(
            itr, paths)

        with torch.no_grad():
            loss_before = self._compute_loss(itr, obs, actions, rewards,
                                             valids, baselines)
            kl_before = self._compute_kl_constraint(obs)

        step_size = self._minibatch_size if self._minibatch_size else len(
            paths)
        for epoch in range(self._max_optimization_epochs):
            shuffled_ids = torch.randperm(len(paths))
            for start in range(0, len(paths), step_size):
                ids = shuffled_ids[start:start + step_size].numpy()
                loss = self._train(itr, obs[ids], actions[ids], rewards[ids],
                                   valids[ids], baselines[ids])
            logger.log('Epoch: {} | Loss: {}'.format(epoch, loss))

        self.baseline.fit(paths)

        self._old_policy.load_state_dict(self.policy.state_dict())

        with torch.no_grad():
            loss_after = self._compute_loss(itr, obs, actions, rewards, valids,
                                            baselines)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        with tabular.prefix(self.policy.name):
            tabular.record('LossBefore', loss_before.item())
            tabular.record('LossAfter', loss_after.item())
            tabular.record('dLoss', loss_before.item() - loss_after.item())
            tabular.record('KLBefore', kl_before.item())
            tabular.record('KL', kl_after.item())
            tabular.record('Entropy', policy_entropy.mean().item())

        return log_performance(itr,
                               TrajectoryBatch.from_trajectory_list(
                                   self.env_spec, paths),
                               discount=self.discount)

    def _train(self, itr, obs, actions, rewards, valids, baselines):
        """Train the algorithm with minibatch.

        Args:
            itr (int): Iteration number.
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.
            valids (list[int]): Array of length of the valid values.
            baselines (torch.Tensor): Value function estimation at each step.

        Returns:
            torch.Tensor: Calculated mean value of loss

        """
        loss = self._compute_loss(itr, obs, actions, rewards, valids,
                                  baselines)

        self._optimizer.zero_grad()
        loss.backward()

        self._optimize(itr, obs, actions, rewards, valids, baselines)

        return loss

    def _compute_loss(self, itr, obs, actions, rewards, valids, baselines):
        """Compute mean value of loss.

        Args:
            itr (int): Iteration number.
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.
            valids (list[int]): Array of length of the valid values.
            baselines (torch.Tensor): Value function estimation at each step.

        Returns:
            torch.Tensor: Calculated mean value of loss

        """
        del itr

        policy_entropies = self._compute_policy_entropy(obs)

        if self._maximum_entropy:
            rewards += self._policy_ent_coeff * policy_entropies

        advantages = compute_advantages(self.discount, self._gae_lambda,
                                        self.max_path_length, baselines,
                                        rewards)
        if self._center_adv:
            valid_transposed_adv = [
                adv[valids > i] for i, adv in enumerate(advantages.T)
            ]
            means = torch.Tensor([adv.mean() for adv in valid_transposed_adv])
            means[torch.isnan(means)] = 0
            varz = torch.Tensor([adv.var() for adv in valid_transposed_adv])
            varz[torch.isnan(varz)] = 1

            advantages = F.batch_norm(advantages, means, varz, eps=self._eps)

        if self._positive_adv:
            advantages -= advantages.min()

        objective = self._compute_objective(advantages, valids, obs, actions,
                                            rewards)

        if self._entropy_regularzied:
            objective += self._policy_ent_coeff * policy_entropies

        valid_objectives = filter_valids(objective, valids)
        return -torch.cat(valid_objectives).mean()

    def _compute_kl_constraint(self, obs):
        """Compute KL divergence.

        Compute the KL divergence between the old policy distribution and
        current policy distribution.

        Args:
            obs (torch.Tensor): Observation from the environment.

        Returns:
            torch.Tensor: Calculated mean KL divergence.

        """
        flat_obs = flatten_batch(obs)
        with torch.no_grad():
            old_dist = self._old_policy.forward(flat_obs)

        new_dist = self.policy.forward(flat_obs)

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs):
        """Compute entropy value of probability distribution.

        Args:
            obs (torch.Tensor): Observation from the environment.

        Returns:
            torch.Tensor: Calculated entropy values given observation

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy.entropy(obs)
        else:
            policy_entropy = self.policy.entropy(obs)

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, valids, obs, actions, rewards):
        """Compute objective value.

        Args:
            advantages (torch.Tensor): Expected rewards over the actions.
            valids (list[int]): Array of length of the valid values.
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.

        Returns:
            torch.Tensor: Calculated objective values

        """
        del valids, rewards
        log_likelihoods = self.policy.log_likelihood(obs, actions)
        return log_likelihoods * advantages

    def _get_baselines(self, path):
        """Get baseline values of the path.

        Args:
            path (dict): collected path experienced by the agent

        Returns:
            torch.Tensor: A 2D vector of calculated baseline with shape(T),
                where T is the path length experienced by the agent.

        """
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        return torch.Tensor(self.baseline.predict(path))

    def _optimize(self, itr, obs, actions, rewards, valids, baselines):
        del itr, valids, obs, actions, rewards, baselines
        self._optimizer.step()

    def process_samples(self, itr, paths):
        """Process sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            tuple:
                * obs (torch.Tensor): The observations of the environment.
                * actions (torch.Tensor): The actions fed to the environment.
                * rewards (torch.Tensor): The acquired rewards.
                * valids (list[int]): Numbers of valid steps in each paths.
                * baselines (torch.Tensor): Value function estimation
                    at each step.

        """
        for path in paths:
            if 'returns' not in path:
                path['returns'] = tensor_utils.discount_cumsum(
                    path['rewards'], self.discount)

        valids = torch.Tensor([len(path['actions']) for path in paths]).int()
        obs = torch.stack([
            pad_to_last(path['observations'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])
        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=self.max_path_length,
                        axis=0) for path in paths
        ])
        rewards = torch.stack([
            pad_to_last(path['rewards'], total_length=self.max_path_length)
            for path in paths
        ])
        baselines = torch.stack([
            pad_to_last(self._get_baselines(path),
                        total_length=self.max_path_length) for path in paths
        ])

        return obs, actions, rewards, valids, baselines
