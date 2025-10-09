import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import magiccube as rubix

MOVES = [
    "U", "U'", "U2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "B", "B'", "B2",
]

INVERSE = {
    "U": "U'", "U'": "U", "U2": "U2",
    "R": "R'", "R'": "R", "R2": "R2",
    "F": "F'", "F'": "F", "F2": "F2",
    "D": "D'", "D'": "D", "D2": "D2",
    "L": "L'", "L'": "L", "L2": "L2",
    "B": "B'", "B'": "B", "B2": "B2",
}

# Solved facelet string in your given format (Y,R,G,O,B,W faces in that order)
SOLVED_3x3 = "YYYYYYYYYRRRRRRRRRGGGGGGGGGOOOOOOOOOBBBBBBBBBWWWWWWWWW"
# For 2x2, each face has 4 stickers, keep the same color order: Y R G O B W
SOLVED_2x2 = "YYYYRRRRGGGGOOOOBBBBWWWW"

# Color vocabulary (fixed order)
COLOR_TO_IDX = {"Y":0, "R":1, "G":2, "O":3, "B":4, "W":5}
VOCAB = "YRGOBW"

def facelet_onehot(facelet_str: str) -> np.ndarray:
    """
    Convert a facelet string (len=24 for 2x2, len=54 for 3x3) to one-hot: [len(facelets) * 6]
    """
    n = len(facelet_str)
    out = np.zeros((n, 6), dtype=np.float32)
    for i, ch in enumerate(facelet_str):
        out[i, COLOR_TO_IDX[ch]] = 1.0
    return out.reshape(-1)

class Rubiks2x2Env:
    """
    Minimal 2x2 environment wrapping magiccube to apply moves and compute rewards.
    """
    def __init__(self, max_steps: int = 50, step_penalty: float = -0.01):
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.reset()

    def reset(self, scramble_len: int = 0) -> np.ndarray:
        self.cube = rubix.Cube(2, SOLVED_2x2)
        self.steps = 0
        self.last_move: Optional[str] = None

        # Apply a random scramble
        if scramble_len > 0:
            seq = self._random_scramble(scramble_len)
            self._apply_moves(seq)
            # After scramble, clear last_move to not block first inverse arbitrarily
            self.last_move = None

        return facelet_onehot(self.cube.get())

    def _apply_moves(self, seq: List[str]):
        for mv in seq:
            self.cube.rotate(mv)

    def _random_scramble(self, n: int) -> List[str]:
        seq = []
        prev = None
        for _ in range(n):
            mv = random.choice(MOVES)
            # avoid immediate inverse of previous to keep scrambles effective
            while prev is not None and mv == INVERSE.get(prev, None):
                mv = random.choice(MOVES)
            seq.append(mv)
            prev = mv
        return seq

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        mv = MOVES[action_idx]

        # Optional: avoid immediate inverse of last move during training to reduce backtracking
        if self.last_move is not None and mv == INVERSE[self.last_move]:
            # small penalty for a "wasted" inverse; still execute to keep MDP simple
            penalty = -0.02
        else:
            penalty = 0.0

        self.cube.rotate(mv)
        self.last_move = mv

        solved = (self.cube.get() == SOLVED_2x2)
        if solved:
            reward = 1.0
            done = True
        else:
            reward = self.step_penalty + penalty
            done = (self.steps >= self.max_steps)

        return facelet_onehot(self.cube.get()), reward, done, {"solved": solved}

# ==============================
# DQN Agent
# ==============================

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        action = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(1)
        next_state = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
        reward = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, n_actions: int = len(MOVES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.98,
        tau: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.policy = QNet(input_dim, 512, n_actions).to(device)
        self.target = QNet(input_dim, 512, n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s)
            return int(q.argmax(dim=1).item())

    def update(self, batch: Tuple[torch.Tensor, ...]):
        state, action, next_state, reward, done = [t.to(self.device) for t in batch]
        q = self.policy(state).gather(1, action)  # [B,1]

        with torch.no_grad():
            # Double DQN target
            next_actions = self.policy(next_state).argmax(dim=1, keepdim=True)
            next_q = self.target(next_state).gather(1, next_actions)
            target = reward + (1.0 - done) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()

        # Soft update target
        with torch.no_grad():
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        return float(loss.item())

def evaluate(env: Rubiks2x2Env, agent: DQNAgent, episodes: int, scramble_len: int) -> Tuple[float, float]:
    solves = 0
    steps_list = []
    for _ in range(episodes):
        s = env.reset(scramble_len)
        for t in range(env.max_steps):
            a = agent.act(s, eps=0.0)
            s, r, done, info = env.step(a)
            if done:
                if info["solved"]:
                    solves += 1
                    steps_list.append(t + 1)
                break
    success = solves / episodes
    avg_steps = (sum(steps_list) / len(steps_list)) if steps_list else 0.0
    return success, avg_steps

def train(
    total_epochs: int = 8,
    episodes_per_epoch: int = 4_000,
    batch_size: int = 256,
    start_eps: float = 1.0,
    end_eps: float = 0.05,
    warmup: int = 5_000,
    scramble_schedule: List[int] = None,
    seed: int = 42,
    save_path: str = "dqn_2x2.pt"
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = Rubiks2x2Env(max_steps=60, step_penalty=-0.01)
    input_dim = 24 * 6  # 2x2 has 24 facelets, 6 colors -> one-hot
    agent = DQNAgent(input_dim=input_dim, n_actions=len(MOVES))

    buffer = ReplayBuffer(300_000)
    global_step = 0

    if scramble_schedule is None:
        # Curriculum: epochs over scramble lengths
        scramble_schedule = [1, 2, 3, 4, 5, 6, 7, 8]

    for epoch, scramble_len in enumerate(scramble_schedule[:total_epochs], 1):
        print(f"\n=== Epoch {epoch}/{total_epochs} | Scramble length = {scramble_len} ===")
        eps_decay_steps = episodes_per_epoch * 0.8
        for ep in range(1, episodes_per_epoch + 1):
            # linear epsilon decay inside each epoch
            frac = min(1.0, max(0.0, (eps_decay_steps - ep) / eps_decay_steps))
            eps = end_eps + (start_eps - end_eps) * frac

            s = env.reset(scramble_len)
            done = False
            step_in_ep = 0

            while not done:
                a = agent.act(s, eps)
                ns, r, done, info = env.step(a)

                buffer.push(s, a, ns, r, float(done))
                s = ns
                global_step += 1
                step_in_ep += 1

                # learn
                if len(buffer) >= max(batch_size, warmup):
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch)

                if done:
                    break

            if ep % 500 == 0:
                succ, avg_steps = evaluate(env, agent, episodes=200, scramble_len=scramble_len)
                print(f"  ep={ep:05d}  ε={eps:.3f}  success={succ*100:.1f}%  avg_steps={avg_steps:.1f}  buffer={len(buffer)}")

        # checkpoint per epoch
        torch.save({"model": agent.policy.state_dict()}, save_path)
        succ, avg_steps = evaluate(env, agent, episodes=500, scramble_len=scramble_len)
        print(f"[EVAL] scramble={scramble_len}  success={succ*100:.1f}%  avg_steps={avg_steps:.1f}  saved:{save_path}")

    print("Training done. Final checkpoint saved:", save_path)
    return save_path


def solve_with_agent(state_str: str, ckpt_path: str, max_moves: int = 60) -> List[str]:
    """
    Given a 2x2 state string, load agent and greedily solve.
    """
    env = Rubiks2x2Env(max_steps=max_moves)
    agent = DQNAgent(input_dim=24*6, n_actions=len(MOVES))
    sd = torch.load(ckpt_path, map_location=agent.device)
    agent.policy.load_state_dict(sd["model"])
    agent.target.load_state_dict(sd["model"])

    cube = rubix.Cube(2, state_str)
    if cube.get() == SOLVED_2x2:
        return []

    seq = []
    last_move = None
    for _ in range(max_moves):
        s = facelet_onehot(cube.get())
        a = agent.act(s, eps=0.0)
        mv = MOVES[a]
        # Optional: avoid immediate inverse at inference too
        if last_move is not None and mv == INVERSE[last_move]:
            # pick next-best action (tie-breaker)
            with torch.no_grad():
                q = agent.policy(torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0))[0]
                q[a] = -1e9
                a = int(torch.argmax(q).item())
                mv = MOVES[a]

        cube.rotate(mv)
        seq.append(mv)
        last_move = mv
        if cube.get() == SOLVED_2x2:
            break
    return seq


if __name__ == "__main__":
    ckpt = train(
        total_epochs=1,          # increase for better performance
        episodes_per_epoch=300, # tune based on time/compute
        batch_size=64,
        warmup=300,
        scramble_schedule=[1,2,3,4,5,6,7,8],
        save_path="dqn_2x2.pt"
    )

    demo_env = Rubiks2x2Env()
    s = demo_env.reset(scramble_len=5)
    scrambled = demo_env.cube.get()
    print("Scrambled:", scrambled)

    seq = solve_with_agent(scrambled, ckpt_path=ckpt, max_moves=60)
    print("Agent solution:", " ".join(seq))

    test_cube = rubix.Cube(2, scrambled)
    for mv in seq:
        test_cube.rotate(mv)
    print("Solved?", test_cube.get() == SOLVED_2x2)
