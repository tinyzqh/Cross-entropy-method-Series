import gym
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import gym.spaces as spaces
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Independent
from torch.distributions import Normal
from typing import Iterable, Optional

# ================================================================
# Reference:
# 1. https://gist.github.com/kashif/5dfa12d80402c559e060d567ea352c06#file-cem-py
# 2. https://github.com/zuoxingdong/lagom/tree/master/baselines/cem
# ================================================================

def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    r"""Applies orthogonal initialization for the parameters of a given module.

    Args:
        module (nn.Module): A module to apply orthogonal initialization over its parameters.
        nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
            is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored.
            Default: ``None``
        weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
            :attr:`nonlinearity` is not ``None``. Default: 1.0
        constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0

    .. note::

        Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
        nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.

    Example::

        # >>> a = nn.Linear(2, 3)
        # >>> ortho_init(a)

    """
    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    else:
        gain = weight_scale

    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
        for name, param in module.named_parameters():
            if 'weight_' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias_' in name:
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, constant_bias)


def tensorify(x, device):
    if torch.is_tensor(x):
        if str(x.device) != str(device):
            x = x.to(device)
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    else:
        return torch.from_numpy(np.asarray(x)).float().to(device)


def numpify(x, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x

def do_episode(policy, env, num_steps, discount=1.0, render=False):
    disc_total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = policy.choose_action(ob)
        (ob, reward, done, _info) = env.step(a)
        disc_total_rew += reward * discount**t
        if render and t%3==0:
            env.render()
        if done: break
    return disc_total_rew

def noisy_evaluation(data):
    # policy = make_policy(theta)
    theta, args = data
    policy = Agent(env=env, args=args).to(args.device)
    policy.from_vec(tensorify(theta, args.device))
    reward = do_episode(policy, env, args.NumSteps)
    return reward


class Module(nn.Module):
    r"""Wrap PyTorch nn.module to provide more helper functions. """

    def __init__(self, **kwargs):
        super().__init__()

        for key, val in kwargs.items():
            self.__setattr__(key, val)

    @property
    def num_params(self):
        r"""Returns the total number of parameters in the neural network. """
        return sum(param.numel() for param in self.parameters())

    @property
    def num_trainable_params(self):
        r"""Returns the total number of trainable parameters in the neural network."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @property
    def num_untrainable_params(self):
        r"""Returns the total number of untrainable parameters in the neural network. """
        return sum(param.numel() for param in self.parameters() if not param.requires_grad)

    # def to_vec(self):
    #     r"""Returns the network parameters as a single flattened vector. """
    #     return parameters_to_vector(parameters=self.parameters())

    def from_vec(self, x):
        r"""Set the network parameters from a single flattened vector.

        Args:
            x (Tensor): A single flattened vector of the network parameters with consistent size.
        """
        self.vector_to_parameters(vec=x, parameters=self.parameters())

    def _check_param_device(self, param: torch.Tensor, old_param_device: Optional[int]) -> int:
        r"""This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        Args:
            param ([Tensor]): a Tensor of a parameter of a model
            old_param_device (int): the device where the first parameter of a
                                    model is allocated.

        Returns:
            old_param_device (int): report device for the first time
        """

        # Meet the first parameter
        if old_param_device is None:
            old_param_device = param.get_device() if param.is_cuda else -1
        else:
            warn = False
            if param.is_cuda:  # Check if in same GPU
                warn = (param.get_device() != old_param_device)
            else:  # Check if in CPU
                warn = (old_param_device != -1)
            if warn:
                raise TypeError('Found two parameters on different devices, this is currently not supported.')
        return old_param_device

    def vector_to_parameters(self, vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
        r"""Convert one vector to the parameters

        Args:
            vec (Tensor): a single vector represents the parameters of a model.
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.
        """
        # Ensure vec of type Tensor
        if not isinstance(vec, torch.Tensor):
            raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
        # Flag for the device where the parameter is located
        param_device = None

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = self._check_param_device(param, param_device)

            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = vec[pointer:pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param

    def save(self, f):
        r"""Save the network parameters to a file.

        It complies with the `recommended approach for saving a model in PyTorch documentation`_.

        .. note::
            It uses the highest pickle protocol to serialize the network parameters.

        Args:
            f (str): file path.

        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        import pickle
        torch.save(obj=self.state_dict(), f=f, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, f):
        r"""Load the network parameters from a file.

        It complies with the `recommended approach for saving a model in PyTorch documentation`_.

        Args:
            f (str): file path.

        .. _recommended approach for saving a model in PyTorch documentation:
            https://pytorch.org/docs/master/notes/serialization.html#best-practices
        """
        self.load_state_dict(torch.load(f))


class CategoricalHead(Module):
    r"""Defines a module for a Categorical (discrete) action distribution.

    Example:

        # >>> import torch
        # >>> action_head = CategoricalHead(30, 4, 'cpu')
        # >>> action_head(torch.randn(2, 30))
        Categorical(probs: torch.Size([2, 4]))

    Args:
        feature_dim (int): number of input features
        num_action (int): number of discrete actions
        device (torch.device): PyTorch device
        **kwargs: keyword arguments for more specifications.

    """

    def __init__(self, feature_dim, num_action, device, **kwargs):
        super().__init__(**kwargs)

        self.feature_dim = feature_dim
        self.num_action = num_action
        self.device = device

        self.action_head = nn.Linear(self.feature_dim, self.num_action)
        # weight_scale=0.01 -> uniformly distributed
        ortho_init(self.action_head, weight_scale=0.01, constant_bias=0.0)

        self.to(self.device)

    def forward(self, x):
        action_score = self.action_head(x)
        action_prob = F.softmax(action_score, dim=-1)
        action_dist = Categorical(probs=action_prob)
        return action_dist


class DiagGaussianHead(Module):
    r"""Defines a module for a diagonal Gaussian (continuous) action distribution which
    the standard deviation is state independent.

    The network outputs the mean :math:`\mu(x)` and the state independent logarithm of standard
    deviation :math:`\log\sigma` (allowing to optimize in log-space, i.e. both negative and positive).

    The standard deviation is obtained by applying exponential function :math:`\exp(x)`.

    Example:

        # >>> import torch
        # >>> action_head = DiagGaussianHead(10, 4, 'cpu', 0.45)
        # >>> action_dist = action_head(torch.randn(2, 10))
        # >>> action_dist.base_dist
        # Normal(loc: torch.Size([2, 4]), scale: torch.Size([2, 4]))
        # >>> action_dist.base_dist.stddev
        # tensor([[0.4500, 0.4500, 0.4500, 0.4500],
        #         [0.4500, 0.4500, 0.4500, 0.4500]], grad_fn=<ExpBackward>)

    Args:
        feature_dim (int): number of input features
        action_dim (int): flat dimension of actions
        device (torch.device): PyTorch device
        std0 (float): initial standard deviation
        **kwargs: keyword arguments for more specifications.

    """

    def __init__(self, feature_dim, action_dim, device, std0, **kwargs):
        super().__init__(**kwargs)
        assert std0 > 0
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device
        self.std0 = std0

        self.mean_head = nn.Linear(self.feature_dim, self.action_dim)
        # 0.01 -> almost zeros initially
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logstd_head = nn.Parameter(torch.full((self.action_dim,), math.log(std0)))

        self.to(self.device)

    def forward(self, x):
        mean = self.mean_head(x)
        logstd = self.logstd_head.expand_as(mean)
        std = torch.exp(logstd)
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        return action_dist


class LinearSchedule(object):
    r"""A linear scheduling from an initial to a final value over a certain timesteps, then the final
    value is fixed constantly afterwards.

    .. note::

        This could be useful for following use cases:

        * Decay of epsilon-greedy: initialized with :math:`1.0` and keep with :attr:`start` time steps, then linearly
          decay to :attr:`final` over :attr:`N` time steps, and then fixed constantly as :attr:`final` afterwards.
        * Beta parameter in prioritized experience replay.

        Note that for learning rate decay, one should use PyTorch ``optim.lr_scheduler`` instead.

    Example:

        # >>> scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=0)
        # >>> [scheduler(i) for i in range(6)]
        [1.0, 0.7, 0.4, 0.1, 0.1, 0.1]

    Args:
        initial (float): initial value
        final (float): final value
        N (int): number of scheduling timesteps
        start (int, optional): the timestep to start the scheduling. Default: 0

    """

    def __init__(self, initial, final, N, start=0):
        assert N > 0, f'expected N as positive integer, got {N}'
        assert start >= 0, f'expected start as non-negative integer, got {start}'

        self.initial = initial
        self.final = final
        self.N = N
        self.start = start

        self.x = None

    def __call__(self, x):
        r"""Returns the current value of the scheduling.

        Args:
            x (int): the current timestep.

        Returns:
            float: current value of the scheduling.
        """
        assert isinstance(x, int) and x >= 0, f'expected as a non-negative integer, got {x}'

        if x == 0 or x < self.start:
            self.x = self.initial
        elif x >= self.start + self.N:
            self.x = self.final
        else:  # scheduling over N steps
            delta = self.final - self.initial
            ratio = (x - self.start) / self.N
            self.x = self.initial + ratio * delta
        return self.x

    def get_current(self):
        return self.x


class Agent(Module):
    def __init__(self, env, args, **kwargs):
        super().__init__()
        self.env = env
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.feature_layers = self.make_fc(input_dim=spaces.flatdim(self.ob_space), hidden_sizes=[64, 64])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in [64, 64]])

        feature_dim = 64
        if isinstance(env.action_space, spaces.Discrete):
            self.action_head = CategoricalHead(feature_dim, env.action_space.n, device, **kwargs)
        elif isinstance(env.action_space, spaces.Box):
            self.action_head = DiagGaussianHead(feature_dim, spaces.flatdim(env.action_space), device, std0=1.0, **kwargs)
        else:
            raise NotImplementedError

    def choose_action(self, x, **kwargs):
        obs = tensorify(x, device=args.device)
        for layer, layer_norm in zip(self.feature_layers, self.layer_norm):
            obs = layer_norm(F.relu(layer(obs)))

        action_dist = self.action_head(obs)
        action = action_dist.sample()
        out = numpify(action, self.ac_space.dtype)# .reshape(-1, self.ac_space.shape[0])
        # out = out.squeeze(0)
        return out

    def make_fc(self, input_dim, hidden_sizes):
        r"""Returns a ModuleList of fully connected layers.

        .. note::

            All submodules can be automatically tracked because it uses nn.ModuleList. One can
            use this function to generate parameters in :class:`BaseNetwork`.

        Example::

            # >>> make_fc(3, [4, 5, 6])
            ModuleList(
              (0): Linear(in_features=3, out_features=4, bias=True)
              (1): Linear(in_features=4, out_features=5, bias=True)
              (2): Linear(in_features=5, out_features=6, bias=True)
            )

        Args:
            input_dim (int): input dimension in the first fully connected layer.
            hidden_sizes (list): a list of hidden sizes, each for one fully connected layer.

        Returns:
            nn.ModuleList: A ModuleList of fully connected layers.
        """
        assert isinstance(hidden_sizes, list), f'expected list, got {type(hidden_sizes)}'

        hidden_sizes = [input_dim] + hidden_sizes

        fc = []
        for in_features, out_features in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            fc.append(nn.Linear(in_features=in_features, out_features=out_features))

        fc = nn.ModuleList(fc)

        return fc


class CEM(object):
    def __init__(self, args, agent):
        super().__init__()
        self.args = args
        self.agent = agent
        self.dim_theta = agent.num_params
        self.theta_mean = np.zeros(self.dim_theta) # Initialize mean and standard deviation
        self.theta_std = np.ones(self.dim_theta)
        self.rng = np.random.RandomState(seed=args.seed)
        self.noise_scheduler = LinearSchedule(*args.noise_scheduler)
        self.n_elite = int(args.BatchSize * args.Elite_Frac)
        self.iter = 0

        self.mean_rewards = []
        self.max_rewards = []

    def ask(self):
        "return solutions: the parameter of neural network"
        extra_noise = self.noise_scheduler(self.iter)
        # Sample parameter vectors
        solutions = self.rng.multivariate_normal(mean=self.theta_mean, cov=np.diag(np.array(self.theta_std**2) + extra_noise), size=self.args.BatchSize)
        return solutions

    def tell(self, solutions, rewards):
        "Update the theta_mean and theta_std Using solutions rewards"
        elite_inds = rewards.argsort()[-self.n_elite:]
        elite_thetas = solutions[elite_inds]

        # Update theta_mean, theta_std
        self.theta_mean = elite_thetas.mean(axis=0)
        self.theta_std = elite_thetas.std(axis=0)

        self.iter += 1

        print("iteration {}. mean f: {}. max f: {}".format(itr, np.mean(rewards), np.max(rewards)))
        self.mean_rewards.append(np.mean(rewards))
        self.max_rewards.append(np.max(rewards))

    def save_results(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (20, 8)})
        plt.subplot(1, 2, 1)
        sns.tsplot(time=list(range(len(self.mean_rewards))), data=self.mean_rewards, color='m', condition='Mean Reward', err_style='ci_band', linestyle='-.', linewidth=2.5, estimator=np.median)
        plt.ylabel("Mean Reward")
        plt.xlabel("Iter")
        plt.title(self.args.EnvId)
        plt.subplot(1, 2, 2)
        sns.tsplot(time=list(range(len(self.max_rewards))), data=self.max_rewards, color='b', condition='Max Reward', err_style='ci_band', linestyle='--', linewidth=2.5, estimator=np.median)
        plt.ylabel("Max Reward")
        plt.xlabel("Iter")
        plt.title(self.args.EnvId)
        plt.savefig('./' + self.args.EnvId + 'Figure_1.png')
        plt.close("all")


def make_env(args):
    env = gym.make(args.EnvId)
    env.seed(seed=args.seed)
    env.observation_space.seed(args.seed)
    env.action_space.seed(args.seed)
    if args.clip_action and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The parameter of CEM')
    # ['Acrobot-v1', 'BipedalWalker-v2', 'Pendulum-v0', 'LunarLanderContinuous-v2', 'CartPole-v0']
    # [ Ant-v2, HalfCheetah-v2, Humanoid-v2]
    parser.add_argument("--EnvId", type=str, help="environment name", default="HalfCheetah-v2")
    parser.add_argument("--NumSteps", type=int, help="maximum length of episode", default=500)
    parser.add_argument("--NumIter", type=int, help="number of iterations of cross entropy method", default=1000)
    parser.add_argument("--BatchSize", type=int, help="number of samples per batch", default=25)
    parser.add_argument("--Elite_Frac", type=float, help="fraction of samples used as elite set", default=0.2)
    parser.add_argument("--extra_std", type=float, help="", default=2.0)
    parser.add_argument("--extra_decay_time", type=int, help="", default=10)
    parser.add_argument("--noise_scheduler", type=list, help="[initial, final, N, start]", default=[0.01, 0.001, 400, 0])
    parser.add_argument("--clip_action", type=bool, help="if clip action", default=True)
    parser.add_argument("--seed", type=int, help="random seed", default=1)
    parser.add_argument('--sample_discount', type=float, default=1, help='discount rate to compute return used in sample')
    parser.add_argument('--discount', type=float, default=0.90, help='discount rate to compute return used in evaluate')
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    print("env is {}".format(args.EnvId))
    # Task settings:
    env = make_env(args)

    agent = Agent(env=env, args=args).to(args.device)

    cem = CEM(args=args, agent=agent)

    for itr in range(args.NumIter):
        # Sample parameter vectors
        thetas = cem.ask()
        data = [(theta, args) for theta in thetas]
        rewards = np.array(list(map(noisy_evaluation, data)))

        cem.tell(solutions=thetas, rewards=rewards)

        agent.from_vec(tensorify(cem.theta_mean, args.device))
        do_episode(agent, env, args.NumSteps, discount=0.90, render=False)
        if itr % 10 == 0 and itr != 0:
            cem.save_results()
