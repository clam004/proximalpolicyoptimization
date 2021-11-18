
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal  


class PPO:

    def __init__(self, env, actor, critic):
        
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self._init_hyperparameters()

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = actor 
        self.critic = critic 

        # actions will be sampled from multivariatenormal distribution 
        # mean parameterized by the actor policy, with std = 0.5 during training
        # but during testing will be deterministic

        # standard deviation does not need to be 0.5
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var) # Sigma 

        # initialize optimizers, point them to the parameters of our networks
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):

        # Default value for hyperparameters, will need to change later
        self.timesteps_per_batch = 6000       # timesteps per batch
        self.max_timesteps_per_episode = 2000 # timesteps per episode
        self.gamma = 0.99 # discount factor

        # number of epochs is chosen arbitrarily to be 5 
        self.n_updates_per_iteration = 5

        # the PPO paper recommends a epsilon of 0.2
        self.clip = 0.2

        # set learning rate
        self.lr = 0.001

    def get_action(self, obs):

        """
        both the actor and critic have been initialized
        actor was initialized to take obs, return dimensions in action space
        critic was initialized to take obs, return 1 dim, for value  
        """

        # Query the actor network the vector returned is a vector of means 
        # to parameterized the multi-variate normal distribution
        mean = self.actor(obs)
        dist = MultivariateNormal(mean,self.cov_mat)

        # use the mean vector to sample from the multi-variate normal distribution
        # returning the sampeld action and log probability of the action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # we want to detach both tensors from the computational graph
        # Our computational graph will start later down the line 
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        """
        calculate sum of future discounted rewards, G's, from the
        list of rewards per batch, shape (num timesteps per episode)
        """
        batch_rtgs = []
        total_rtgs = []

        # Iterate through each episode's rewards backwards to maintain
        # same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # backwards accumulator
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma 
                batch_rtgs.insert(0, discounted_reward) # add to left side of list
            total_rtgs.append(discounted_reward)
        # convert the G's into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        
        return batch_rtgs, total_rtgs

    def rollout(self):

        """
        observations: (number of timesteps per batch, dimension of observation)
        actions: (number of timesteps per batch, dimension of action)
        log probabilities: (number of timesteps per batch)
        rewards: (number of episodes, number of timesteps per episode)
        reward-to-go’s: (G, sum of future discounted rewards)
        batch lengths: (number of episodes)
        """

        # Batch data
        batch_obs = []        # batch observations
        batch_acts = []       # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []       # batch rewards
        batch_lens = []       # episodic lengths in batch

        t = 0 # number of timesteps run so far this batch
        while t < self.timesteps_per_batch:

            # Rewards this episode
            ep_rews = []

            obs = self.env.reset() # start from beginning of new episode
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran in this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                # policy decides from state, the next action
                # get the next state, reward, bool for if done, and metadata
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Collect reward, action, and log prob
                # at episode level
                ep_rews.append(rew)
                # at batch level
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # done is True when episode in terminal state or final goal achived
                if done:
                    break

            # Collect batch level episodic lengths and rewards
            batch_lens.append(ep_t + 1) # + 1 because episode_timestep starts at 0
            batch_rews.append(ep_rews) # list of lists of rewards

        # Convert these lists to tensors, we will need them in this form for the
        # computational graph
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # ALG STEP 4 - calculate G's
        # G's = batch rewards-to-go, sum of future discounted rewards
        batch_rtgs, total_rtgs = self.compute_rtgs(batch_rews) # G's

        # return the batch of episodes
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, total_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):

        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Query actor network for the log_prob of each action taken because of batch_obs
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def learn(self, total_timesteps, 
              track_progress = True, 
              savename = '', savepath = '../Models/'):

        """
        There are 3 levels of accumulation of state, action, rewards tuples:
        There is each training run (ie learn()), each batch, and each episode in a batch

        An iteration k, runs a batch of episodes, aka rollouts.
        """

        best_return = -np.inf
        t_so_far = 0 # timesteps rolled out so far, all batches

        # ALG STEP 2 - main learning loop, for iterations k = 0,1,2 .. do
        while t_so_far < total_timesteps:

            # ALG STEP 3 = collect set of trajectories D_k = {T_i} by running policy
            # batch_log_probs is used for pi_theta_k(a_t|s_t)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, total_rtgs, batch_lens = self.rollout()

            # Calculate V_{phi,k} for this iteration, not for this epoch
            V_k, _ = self.evaluate(batch_obs, batch_acts)

            # ALG STEP 5 -Calculate advantage for this k-th iteration under the while loop
            A_k = batch_rtgs - V_k.detach()

            # Normalize advantages to stabilized training even more
            A_k = (A_k - A_k.mean())/(A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):

                # Calculate V_phi, pi_theta(a_t|s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                """
                Importance sampling
                recall that the importance ratio pi_theta(a_t|s_t)/pi_theta_k(a_t|s_t)
                is inside the PPO-Clip objective of step 6

                recall that log(a/b) = log(a) - log(b)
                recall that e^(ln(a)) =  a

                this means that a/b = e^(log(a) - log(b))
                which means e^(curr_log_probs - batch_log_probs) = curr_probs/batch_probs

                ratios = curr_probs/batch_probs
                """

                # note that curr_log_probs is not detached
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # The PPO-Clip objective has a min function between two surrogate losses
                # surr1 is the first surrogate loss in the min func
                # surr2 is the 2nd surrogate loss in the min func
                surr1 = ratios * A_k

                # ratios get cliped between 0.8 and 1.2 if epsilon = 0.2
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k 

                # the entire actor loss as a minimum of two surrogate losses. As the 2 summations 
                # in the pseudocode, suggest, this is across each timestep of each trajectory
                # of this particular epoch of this particular iteration. The minus sign
                # is because we actually want to maximize this objective, ie the advantage
                actor_loss = (-torch.min(surr1, surr2)).mean()

                """
                We will be doing 2 backprops on our computational graph
                both the actor and critic loss computation graphs converge a bit up the graph, 
                for example, batch_rtgs is used to calculate both A_k in the surrogate actor loss
                and also the critic MSE loss. 
                we’ll need to add a retain_graph=True to 
                backward for either actor or critic (depends on which we back propagate on first). 
                Otherwise, we’ll get an error saying that the buffers have already been 
                freed when trying to run backward through the graph a second time.
                """
                # Calculate gradients, do backprop on actor network
                self.actor_optim.zero_grad() # clean gradients from optimizer
                actor_loss.backward(retain_graph=True) # calculate gradients wrt actor objective
                self.actor_optim.step() # apply gradients to update the actor parameters

                # the value function is updated by minimizing a mean-squared error loss 
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients, do backprop on critic network
                self.critic_optim.zero_grad() # clean gradients from optimizer
                critic_loss.backward() # calculate gradients wrt critic objective
                self.critic_optim.step() # apply gradients to update the critic parameters

            # increment t_so_far with how many timesteps we collected this batch iteration
            t_so_far += np.sum(batch_lens)

            #last_return = float(batch_rtgs.mean().data)
            last_return = np.mean(total_rtgs)

            if (last_return > best_return):
                best_return = last_return
                print('new best_return', best_return)
                torch.save(self.actor.state_dict(), savepath+savename+'ppo_actor.pth')
                torch.save(self.critic.state_dict(), savepath+savename+'ppo_critic.pth')

            if track_progress:
                print('actor_loss', actor_loss.data, 
                      'last return', last_return,
                      'all batch timesteps rolled out so far', t_so_far, '/', total_timesteps,
                      )
            
 










