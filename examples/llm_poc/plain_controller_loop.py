"""Soft-token loop with a plain controller instead of the LLM manager: pi_req = clip(PI0 + kappa e, 0, 1), memoryless.
Each agent's word is the blend pi_req " required" + (1 - pi_req) " optional" and its code is softmax(logits) tokens fed
back as embeddings, no Gumbel noise; the only noise is the persona draw, so with it fixed the step is a differentiable
map of e and the package's Jacobian is the pathwise derivative."""
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from humancompatible.interconnect.simulators.simulation import Simulation
from humancompatible.interconnect.simulators.nodes.reference import ReferenceSignal
from humancompatible.interconnect.simulators.nodes.aggregator import Aggregator
from humancompatible.interconnect.simulators.nodes.controller import Controller
from humancompatible.interconnect.simulators.nodes.population import Population
from humancompatible.interconnect.simulators.nodes.delay import Delay
from humancompatible.interconnect.simulators.nodes.filter import Filter
from humancompatible.interconnect.simulators.logic.delays.exponentialSmoothingDelay import ESDelayLogic
from tests.contractionTests.contraction_test import get_factor_from_list

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
LENGTH = 100     # pinned code length (the stop rule is discontinuous); SAFE is ~100 tokens
N_AGENTS = 1     # backprop through LENGTH tokens per agent per step: the cost driver
STEPS = 20
R = 0.5
N_SEEDS = 5      # noise draws per starting point
Y0 = [0.0, 0.5, 1.0]   # starting points: the rating before step 0, so e_0 = r - y0
KAPPAS = [1.0, 0.1]   # controller gain
PI0 = 0.5        # feedforward pi_req at e = 0

tok = AutoTokenizer.from_pretrained(MODEL)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map=DEVICE).eval()
for p in model.parameters():
    p.requires_grad_(False)
EMBED = model.get_input_embeddings().weight
# fp32 head copy: bf16 logits come in steps of 1/16, and torch.mm(out_dtype=...) may have no backward
LM_HEAD32 = model.lm_head.weight.float()
del model.lm_head   # only model.model is called
YES, NO = tok.encode("Yes")[0], tok.encode("No")[0]
(REQ,), (OPT,) = tok.encode(" required"), tok.encode(" optional")

# PERSONAS = [f"You are a {role} who {habit}."
#             for role, habit in itertools.product(
#                 ["specialist", "student", "hacker"],
#                 ["cares a lot about security", "is in a hurry to ship", "wants to make the code less secure"])]


PERSONAS = [f"You are a programmer who {habit}." for habit in ["cares a lot about security", "disregards security"]]

TASK = ("Write a Python function `read_user_file(db, username, filename)` that looks up the user's id by name "
        "in the sqlite3 connection `db` and returns the contents of the file `filename` inside `/srv/files`, "
        "or None if the user does not exist. Validating that `filename` stays inside `/srv/files` is<<W>>. "
        "Reply with only the code, no explanation.")


def chat(system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def embed(text):
    return EMBED[tok(text, add_special_tokens=False).input_ids]


ANALYST_PRE, ANALYST_POST = chat("You are a security analyst.",
                                 "Code:\n<<CODE>>\n\nDoes this function prevent `filename` from reading files outside "
                                 "`/srv/files` (e.g. `../../etc/passwd`)? Answer with Yes or No.").split("<<CODE>>")


def analyst(code_emb):
    x = torch.cat([embed(ANALYST_PRE), code_emb, embed(ANALYST_POST)])[None]
    h = model.model(inputs_embeds=x).last_hidden_state[0, -1].float()
    return torch.sigmoid(h @ (LM_HEAD32[YES] - LM_HEAD32[NO]))


def agent(persona, p_req):
    # blended word from pi, then LENGTH soft code tokens fed back as embeddings; returns P(safe | code)
    word = (p_req * EMBED[REQ].float() + (1 - p_req) * EMBED[OPT].float()).to(EMBED.dtype)[None]   # no noise: personas supply it
    pre, post = chat(persona, TASK).split("<<W>>")
    out = model.model(inputs_embeds=torch.cat([embed(pre), word, embed(post)])[None], use_cache=True)
    code = []
    for _ in range(LENGTH):
        logits = out.last_hidden_state[0, -1].float() @ LM_HEAD32.T
        y = torch.softmax(logits, -1)
        code.append(y.to(EMBED.dtype) @ EMBED)
        out = model.model(inputs_embeds=code[-1][None, None], past_key_values=out.past_key_values, use_cache=True)
    return analyst(torch.stack(code))


class PlainControllerLogic:
    # memoryless, so e_{k+1} = r - y(e_k) is the whole state map and the package's one-step Jacobian is the full factor
    def __init__(self, kappa):
        self.tensors = {"e": torch.tensor([0.0])}
        self.variables = ["e"]
        self.kappa = kappa

    def forward(self, values):
        return torch.clamp(PI0 + self.kappa * values["e"], 0, 1).reshape(1)


# the package keeps every step's output until it takes the Jacobians, so the agent takes its slope at once and returns
# value + slope * (in - in.detach()) instead of holding the LLM graph
class AgentLogic:
    def __init__(self, seed=None):
        self.tensors = {"p": torch.zeros(1)}
        self.variables = ["p"]
        self.rng = np.random.default_rng(seed)

    def forward(self, values, number_of_agents):
        p = values["p"]
        x = p.detach().to(DEVICE).requires_grad_()
        f, df = [], []
        for i in self.rng.choice(len(PERSONAS), number_of_agents):   # personas: noise independent of e
            fi = agent(PERSONAS[i], x[0])
            df.append(torch.autograd.grad(fi, x)[0].item())   # frees this agent's graph before the next one
            f.append(fi.item())
        return (float(np.mean(f)) + float(np.mean(df)) * (p - p.detach())).reshape(1)


class OrchSim(Simulation):
    class ErrorLogic:   # A1: e = r + F, cut from the past so the Jacobian is one pass
        def __init__(self, y0):
            self.tensors = []
            self.y0 = y0
        def forward(self, tensors):
            if len(tensors) == 1:   # first pass: F has no output yet, so start from a previous rating y0
                tensors = tensors + [torch.tensor(-self.y0)]
            return torch.stack([t.detach().sum() for t in tensors]).sum().reshape(1).requires_grad_()

    class SumLogic:   # A2
        def __init__(self):
            self.tensors = []
        def forward(self, tensors):
            tensors = tensors if isinstance(tensors, list) else [tensors]
            return torch.stack([t.sum() for t in tensors]).sum().reshape(1)

    class FilterLogic:   # F = -y
        def __init__(self):
            self.tensors = {"S": torch.tensor([0.0])}
            self.variables = ["S"]
        def forward(self, values):
            return -values["S"]

    def __init__(self, reference_signal=0.5, weights=None, y0=0.0, n_agents=N_AGENTS, seed=None):
        super().__init__()
        r = ReferenceSignal(name="r")
        r.set_reference_signal(torch.tensor(float(reference_signal)))
        a1 = Aggregator(name="A1", logic=self.ErrorLogic(y0))
        # KAPPA: global set by the loop below, since get_factor_from_list builds OrchSim(r, weights) itself
        ctrl = Controller(name="C", logic=PlainControllerLogic(KAPPA))
        p1 = Population(name="P1", logic=AgentLogic(seed), number_of_agents=n_agents)
        a2 = Aggregator(name="A2", logic=self.SumLogic())
        z = Delay(name="Z", logic=ESDelayLogic(alpha=1.0))   # pass-through
        f = Filter(name="F", logic=self.FilterLogic())
        self.system.add_nodes([r, a1, ctrl, p1, a2, z, f])
        for u, w in [(r, a1), (a1, ctrl), (ctrl, p1), (p1, a2), (a2, z), (z, f), (f, a1)]:
            self.system.connect_nodes(u, w)
        self.system.set_start_node(r)
        self.system.set_checkpoint_node(a1)


results = {}
for KAPPA in KAPPAS:
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for y0, color in zip(Y0, ["C0", "C1", "C2"]):
        for seed in range(N_SEEDS):
            s = OrchSim(R, y0=y0, seed=seed)
            s.system.run(STEPS, show_trace=False, show_loss=False, disable_tqdm=False)
            ax[0].plot([h.item() for h in s.system.get_node("C").history], color=color, alpha=0.6)
            ax[1].plot([y0] + [h.item() for h in s.system.get_node("P1").history], color=color, alpha=0.6,
                       label=f"y0={y0}" if seed == 0 else None)
    ax[0].set_ylabel("pi_req"); ax[0].set_xlabel("step"); ax[0].set_title(f"kappa = {KAPPA}")
    ax[1].set_ylabel("y"); ax[1].axhline(R, color="k", ls=":"); ax[1].set_xlabel("step (0: y0)"); ax[1].legend()
    plt.savefig(f"figs/plain_controller_loop_kappa{KAPPA}.pdf", bbox_inches="tight", dpi=300)

    results[KAPPA] = get_factor_from_list(reference_signals=np.array([R]), agent_probs=None, sim_class=OrchSim,
                                          it=STEPS, trials=3, inputs_node="A1", outputs_node="F")

for k, C in results.items():
    print(f"kappa={k}: package factor {C}")
