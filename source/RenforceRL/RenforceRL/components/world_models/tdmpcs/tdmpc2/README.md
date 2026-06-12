# TDMPC2

## Notes

### update

For data integration, we assume data formulated in `[T, B, *D]` when update, the data in replay buffer is ok to be in `[B, T, *D]`

We have following components:

1. Transition NN
    - $z_{t+1} = f(s_t, a_t)$
2. Encoder NN
    - $z_t = f(s_t)$
3. Reward head
    - $r_t = f(s_t, a_t)$
    - [ ? ] $r_t = f(z_{t+1})$
4. Term head
    - $d_t = f(z_{t+1})$
5. Q
    - $q_t = f(z_t, a_t)$
6. Pi
    - $a = f(z_t)$

For loss target:
1. consistency_loss
2. reward_loss
3. value_loss
4. term_loss
5. pi_loss


That's ok for update, however the paralleled planning not support the batch planning.

###

Current td target has a problem:
```
        with torch.no_grad(): 
            td_target = reward[:-1] + self.cfg.gamma * (1-terminated[:-1]) * self.q_network(z_enc[1:], action[1:])
```
We assume the `z_enc` will output `action`, this may work when the sample policy is our policy; however, the policy is under the z while ours is accepting the latent.

Why not PPO update for pi?

## remarks

### pi

There the tdmpc2 should have in model, whatever it sucks.

In the mean time, there is no target encoder, where the consistensy is also part of the loss.

### planner problems

```
# ---------------------------
# Points that still require confirmation
#
# 4) prior_mean_action_fn semantics:
#    - I expect a callable that takes raw obs [B, obs_dim] and returns [B, A].
#      If your policy instead expects latent inputs, either wrap it to accept obs or pass a different callable.
#
# 5) How to handle prior selection when prior_count > K (warmup sequences):
#    - current behavior: use first K prior sequences (deterministic), and fill remaining slots with prior_mean_action_fn if available.
#      If you prefer random sampling from K or sampling with replacement, tell me and I will change to sample randomly.
#
# 6) gamma value for bootstrap:
#    - I fetch getattr(world_model, "gamma", 0.99). Confirm if you want a different default or to pass gamma via cfg.
# ---------------------------
```

## todo list

encoder ema for encoding shifting.

Fully original tdmpc2 method with offline interaction (Test it work or not.)

Model structure specification. (Hope it may work.)

Offline methods like C51 DSAC, as metioned in the fastSAC and fastTD3, they emphisis that it could work?
Does it work? Try it

** Termination will make tdmpc not work. **

** reward scale **
