import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

import ftd.algorithms.modules as m
import ftd.net.auxiliary_pred as aux
from ftd.algorithms.sac import SAC
from ftd.net.image_attention import ImageAttentionSelectorLayers
from segdac.agents.agent import Agent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy
from segdac.data.mdp import MdpData
from tensordict import TensorDict


class FtdActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(
        self,
        actor,
    ):
        super().__init__(actor)

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        obs = mdp_data.data["pixels_transformed"]  # (b, frame_stack, nb_segs, c, h, w)

        test_env = not self.is_exploration_enabled and not self.is_stochasticity_enabled

        if test_env:
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            action = mu
        else:
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            action = pi

        return TensorDict(
            {"unscaled_action": action}, batch_size=torch.Size([action.shape[0]])
        )


class Image_Selector_SAC(SAC, Agent):
    def __init__(
        self,
        obs_shape,
        action_shape,
        discount,
        critic_tau,
        encoder_tau,
        actor_update_freq,
        critic_target_update_freq,
        unsupervised_update_freq,
        unsupervised_update_num,
        unsupervised_update_slow_freq,
        frame_stack,
        channels,
        masked_region_num,
        attention_heads,
        max_grad_norm,
        reward_factor,
        inv_factor,
        fwd_factor,
        reward_accumulate_steps,
        inv_accumulate_steps,
        fwd_accumulate_steps,
        unsupervised_warmup_steps,
        reward_first_sampling,
        num_selector_layers,
        num_filters,
        embed_dim,
        num_shared_layers,
        num_head_layers,
        projection_dim,
        hidden_dim,
        actor_log_std_min,
        actor_log_std_max,
        init_temperature,
        actor_lr,
        actor_beta,
        critic_lr,
        critic_beta,
        alpha_lr,
        alpha_beta,
        selector_lr,
        selector_beta,
        env_action_scaler: TanhEnvActionScaler,
    ):
        Agent.__init__(self, env_action_scaler, None)
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.unsupervised_update_freq = unsupervised_update_freq
        self.unsupervised_update_num = unsupervised_update_num
        self.unsupervised_update_slow_freq = unsupervised_update_slow_freq
        self.stack_num = frame_stack
        self.channels = channels
        self.region_num = masked_region_num
        self.attention_heads = attention_heads
        self.max_grad_norm = max_grad_norm

        self.reward_factor = reward_factor
        self.inverse_factor = inv_factor
        self.forward_factor = fwd_factor
        self.reward_accumulated_steps = reward_accumulate_steps
        self.inv_accumulated_steps = inv_accumulate_steps
        self.fwd_accumulated_steps = fwd_accumulate_steps
        self.unsupervised_warmup_steps = unsupervised_warmup_steps
        self.reward_first_sampling = reward_first_sampling

        selector_layers = ImageAttentionSelectorLayers(
            obs_shape,
            masked_region_num,
            channels,
            frame_stack,
            num_selector_layers,
            num_filters,
            embed_dim,
            attention_heads,
        )
        selector_cnn = m.SelectorCNN(
            selector_layers,
            obs_shape,
            masked_region_num,
            channels,
            frame_stack,
            num_shared_layers,
            num_filters,
        ).cuda()
        head_cnn = m.HeadCNN(
            selector_cnn.out_shape, num_head_layers, num_filters
        ).cuda()
        actor_projection = m.RLProjection(head_cnn.out_shape, projection_dim)
        critic_projection = m.RLProjection(head_cnn.out_shape, projection_dim)
        actor_encoder = m.Encoder(selector_cnn, head_cnn, actor_projection)
        critic_encoder = m.Encoder(selector_cnn, head_cnn, critic_projection)

        actor = m.Actor(
            actor_encoder,
            action_shape,
            hidden_dim,
            actor_log_std_min,
            actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(critic_encoder, action_shape, hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.action_sampling_strategy = FtdActionSamplingStrategy(
            actor=actor,
        )

        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.complete_selector = selector_layers.cuda()
        self.selector_optimizer = torch.optim.Adam(
            self.complete_selector.parameters(),
            lr=selector_lr,
            betas=(selector_beta, 0.999),
        )

        self.reward_predictor = aux.RewardPredictor(
            critic_encoder,
            action_shape,
            reward_accumulate_steps,
            hidden_dim,
            num_filters,
        ).cuda()
        self.reward_predictor_optimizer = torch.optim.Adam(
            self.reward_predictor.parameters(),
            lr=selector_lr,
            betas=(selector_beta, 0.999),
        )

        self.inverse_dynamic_predictor = aux.InverseDynamicPredictor(
            critic_encoder,
            action_shape,
            inv_accumulate_steps,
            hidden_dim,
            num_filters,
        ).cuda()
        self.inverse_dynamic_predictor_optimizer = torch.optim.Adam(
            self.inverse_dynamic_predictor.parameters(),
            lr=selector_lr,
            betas=(selector_beta, 0.999),
        )

        self.forward_dynamic_predictor = aux.ForwardDynamicPredictor(
            critic_encoder,
            action_shape,
            fwd_accumulate_steps,
            hidden_dim,
            num_filters,
        ).cuda()
        self.forward_dynamic_predictor_optimizer = torch.optim.Adam(
            self.forward_dynamic_predictor.parameters(),
            lr=selector_lr,
            betas=(selector_beta, 0.999),
        )

        self.train()
        self.critic_target.train()

    @property
    def actor(self):
        return self.action_sampling_strategy.actor

    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        logs_data = {}

        if (
            env_step % self.unsupervised_update_slow_freq == 0
            and env_step > self.unsupervised_warmup_steps
        ):
            self.unsupervised_update_freq = self.unsupervised_update_freq + 1
            print("Update frequency slow down to ", str(self.unsupervised_update_freq))

        b, frame_stack, nb_segs, c, h, w = train_mdp_data.data[
            "pixels_transformed"
        ].shape
        obs = train_mdp_data.data["pixels_transformed"]
        action = train_mdp_data.data["action"]
        reward = train_mdp_data.next.data["reward"].reshape(-1, 1)
        next_obs = train_mdp_data.next.data["pixels_transformed"]
        not_done = ~train_mdp_data.next.data["done"].reshape(-1, 1)

        if is_time_to_evaluate:
            L = logs_data
        else:
            L = None

        self.update_critic(obs, action, reward, next_obs, not_done, L, env_step)

        if env_step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, env_step)

        if env_step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if (
            env_step % self.unsupervised_update_freq == 0
            and env_step > self.unsupervised_warmup_steps
        ):
            for _ in range(self.unsupervised_update_num):
                if self.reward_factor != 0:
                    self.update_reward_predictor(L, env_step)
                if self.inverse_factor != 0:
                    self.update_inverse_dynamic_predictor(L, env_step)

        return TensorDict(logs_data, batch_size=torch.Size([]))

    def update_reward_predictor(self, L, step):
        reward_predictor_mdp_data = self.replay_buffer.sample().to("cuda")
        obs = reward_predictor_mdp_data.data["pixels_transformed"]
        action = reward_predictor_mdp_data.data["action"]
        reward = reward_predictor_mdp_data.next.data["reward"].reshape(-1, 1)

        concatenated_action = action
        concatenated_reward = reward
        predicted_reward = self.reward_predictor(obs, concatenated_action)
        predict_loss = self.reward_factor * F.mse_loss(
            concatenated_reward, predicted_reward
        )

        self.reward_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(
                self.reward_predictor.parameters(), self.max_grad_norm
            )
        self.reward_predictor_optimizer.step()

        if L is not None:
            L["reward_predictor_loss"] = predict_loss.detach()

    def update_inverse_dynamic_predictor(self, L, step):
        inverse_dynamic_predictor_mdp_data = self.replay_buffer.sample().to("cuda")
        obs = inverse_dynamic_predictor_mdp_data.data["pixels_transformed"]
        action = inverse_dynamic_predictor_mdp_data.data["action"]
        next_obs = inverse_dynamic_predictor_mdp_data.next.data["pixels_transformed"]

        B, _ = action.shape
        previous_actions = action.new_empty(
            (B, 0)
        )  # Since accumulation steps are 1 for FTD

        predicted_action = self.inverse_dynamic_predictor(
            obs, next_obs, previous_actions
        )
        predict_loss = self.inverse_factor * F.mse_loss(action, predicted_action)

        self.inverse_dynamic_predictor_optimizer.zero_grad()
        predict_loss.backward()
        if self.max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(
                self.inverse_dynamic_predictor.parameters(), self.max_grad_norm
            )
        self.inverse_dynamic_predictor_optimizer.step()

        if L is not None:
            L["inverse_dynamic_predictor_loss"] = predict_loss.detach()
