# HAPPO Algorithm Explained

## Overview
**Heterogeneous-Agent Proximal Policy Optimization (HAPPO)** is a multi-agent reinforcement learning algorithm designed for cooperative tasks with heterogeneous agents. It extends PPO to handle agents with different capabilities without requiring parameter sharing.

---

## Key Concepts

### 1. Multi-Agent Advantage Decomposition
The joint advantage can be decomposed sequentially:

$$A_\pi^{i_{1:m}}(s, a^{i_{1:m}}) = \sum_{j=1}^{m} A_\pi^{i_j}(s, a^{i_{1:j-1}}, a^{i_j})$$

This allows evaluating each agent's contribution while accounting for previous agents' actions.

### 2. Sequential Update Scheme
Agents update their policies **one at a time** in a **random order** $i_{1:n}$ (permutation of all agents). Each agent considers the updated policies of agents that came before it in the sequence.

---

## HAPPO Algorithm: Step-by-Step

### **Step 1: Initialization**
- **Actor networks** (policy): $\{\theta_0^i, \forall i \in N\}$ - one per agent
- **Global V-value network** (critic): $\phi_0$ - shared across all agents
- **Replay buffer**: $\mathcal{B}$

### **Step 2: Data Collection**
Run the current joint policy $\pi_{\theta_k} = (\pi_{\theta_k^1}^1, \ldots, \pi_{\theta_k^n}^n)$ in the environment to collect trajectories:

$$\{(s_t, o_t^i, a_t^i, r_t, s_{t+1}, o_{t+1}^i), \forall i \in N, t \in T\}$$

Store transitions in replay buffer $\mathcal{B}$.

### **Step 3: Advantage Estimation**
Compute the **joint advantage function** $\hat{A}(s, a)$ using the global V-value network with **Generalized Advantage Estimation (GAE)**:

$$\hat{A}(s_t, a_t) = \sum_{\ell=0}^{\infty} (\gamma \lambda)^\ell \delta_{t+\ell}$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$

**Key difference from MAPPO**: HAPPO uses a single global critic for all agents, not separate critics.

### **Step 4: Random Agent Permutation**
Draw a **random permutation** $i_{1:n}$ of all $n$ agents. This randomization is crucial for theoretical guarantees.

### **Step 5: Sequential Policy Updates**

Initialize the modified advantage:
$$M^{i_1}(s, a) = \hat{A}(s, a)$$

**For each agent $i_m$ in sequence** ($m = 1, \ldots, n$):

#### 5a. Update Policy Using PPO-Clip Objective
Agent $i_m$ updates its policy parameter $\theta_k^{i_m} \to \theta_{k+1}^{i_m}$ by maximizing:

$$\max_{\theta^{i_m}} \mathbb{E}_{s, a} \left[ \min \left( r(\bar{\pi}^{i_m}) M^{i_{1:m}}(s, a), \text{clip}(r(\bar{\pi}^{i_m}), 1 \pm \epsilon) M^{i_{1:m}}(s, a) \right) \right]$$

where:
- **Probability ratio**: $r(\bar{\pi}^{i_m}) = \frac{\pi_{\theta^{i_m}}^{i_m}(a^{i_m}|o^{i_m})}{\pi_{\theta_k^{i_m}}^{i_m}(a^{i_m}|o^{i_m})}$
- **Clip function**: $\text{clip}(r, 1 \pm \epsilon)$ bounds the ratio to $[1-\epsilon, 1+\epsilon]$
- **Modified advantage**: $M^{i_{1:m}}(s, a)$ accounts for all previous agents' updates

**What the agent network estimates**: The agent network (actor) outputs the policy $\pi_{\theta^{i_m}}^{i_m}(a^{i_m}|o^{i_m})$ - the probability distribution over actions given observations.

#### 5b. Compute Modified Advantage for Next Agent
After updating agent $i_m$, compute the modified advantage for the next agent:

$$M^{i_{1:m+1}}(s, a) = \frac{\pi_{\theta_{k+1}^{i_m}}^{i_m}(a^{i_m}|o^{i_m})}{\pi_{\theta_k^{i_m}}^{i_m}(a^{i_m}|o^{i_m})} M^{i_{1:m}}(s, a)$$

This propagates the importance of the policy change to subsequent agents.

### **Step 6: Critic Update**
After all agents have updated, update the global V-value network to minimize:

$$\phi_{k+1} = \arg\min_\phi \frac{1}{BT} \sum_{b=1}^B \sum_{t=0}^T \left( V_\phi(s_t) - \hat{R}_t \right)^2$$

where $\hat{R}_t$ is the estimated return (e.g., discounted sum of rewards).

**What the critic network estimates**: The critic outputs $V_\phi(s)$ - the expected return from state $s$ under the current joint policy.

### **Step 7: Repeat**
Return to Step 2 for the next iteration.

---

## Key Differences from MAPPO

| Feature | MAPPO | HAPPO |
|---------|-------|-------|
| **Parameter sharing** | Typically uses shared parameters | No parameter sharing (heterogeneous) |
| **Update scheme** | Simultaneous updates | Sequential updates with random order |
| **Advantage modification** | Uses raw advantage $\hat{A}(s,a)$ | Uses modified advantage $M^{i_{1:m}}(s,a)$ |
| **Coordination** | No explicit coordination between agents | Agents account for previous updates via $M^{i_{1:m}}$ |
| **Theoretical guarantees** | No monotonic improvement guarantee | Guaranteed monotonic improvement & Nash equilibrium convergence |
| **Applicability** | Best for homogeneous agents | Designed for heterogeneous agents |

---

## Summary of Neural Network Roles

1. **Agent Networks (Actors)** $\{\theta^i\}$:
   - **Input**: Agent's observation $o^i$
   - **Output**: Policy $\pi_{\theta^i}^i(a^i|o^i)$ (action probabilities)
   - **One network per agent** (heterogeneous)

2. **Critic Network** $\phi$:
   - **Input**: Global state $s$
   - **Output**: Value function $V_\phi(s)$ (expected return)
   - **Single shared network** for all agents

---

## Theoretical Properties

HAPPO guarantees:
1. **Monotonic improvement**: Joint return improves with each iteration
2. **Nash equilibrium convergence**: Converges to a Nash equilibrium policy
3. **No need for parameter sharing**: Works with heterogeneous agents

The sequential update scheme with random agent ordering is crucial for these guarantees.
