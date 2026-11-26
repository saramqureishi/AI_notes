# CS5707 — WEEK 9: LARGE LANGUAGE MODELS — Seminar Answers (detailed)

> Sources: Lecture W9 (all 55 slides, text only; figure slides carry no text), Lab W9 + the three notebooks
> (outputs are from Sara's own Colab runs), and the papers: Vaswani 2017, Merullo 2023, Bsharat 2023,
> Hu 2021 (LoRA), Ziegler 2020. Numbers marked "paper" were checked against the PDF text; calculations were run in Python.
>
> **Honest gaps**
> - I did NOT re-run the labs: they need a Gemini API key and gated models (Gemma, Llama). Lab results are quoted from your saved notebook outputs.
> - The full Week 9 lecture notes and the quiz are not written yet (see end).
> - Each answer has: **the answer**, a **worked example** where useful, a **mark-scheme list** (what an examiner ticks), and **common mistakes**.

---

## Q1. The Transformer: attention only; why it beats RNNs on parallelisation and efficiency

### 1.1 The problem it solved
Before 2017 the best sequence models (translation, language modelling) were **RNNs / LSTMs / GRUs**. They read a sentence **one token at a time**, carrying a hidden
state forward. Two consequences (paper): computation "factors along symbol positions", so the sequential nature **"precludes parallelisation within training
examples"**, and information from early words must survive many steps to reach late words. Vaswani et al. proposed the **Transformer**: a model "based
**solely on attention mechanisms**, dispensing with recurrence and convolutions entirely".

### 1.2 Self-attention step by step
For each token (an embedding vector) the model computes three vectors using three learned weight matrices:
- **Query q** = "what am I looking for?"
- **Key k** = "what do I contain / offer?"
- **Value v** = "what information do I pass on if attended to?"

Then, for the whole sequence at once (matrices Q, K, V):
```
 Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
```
1. `Q K^T` = every query dotted with every key = an **n × n score matrix** (how relevant is token j to token i).
2. Divide by **√d_k** (see 1.4).
3. **softmax** each row → weights that are ≥ 0 and **sum to 1** (the "attention matrix", lecture slides 16–17).
4. Multiply by V → each token's new vector is a **weighted mix of all tokens' values**.

**Worked example (verified in Python).** 3 tokens, d = 2. Embeddings X = [[1,0],[0,1],[1,1]]; take W_q = W_k = identity, W_v = [[1,2],[3,4]], so
V = X·W_v = [[1,2],[3,4],[4,6]].
```
 scores QK^T / sqrt(2)            attention weights (softmax per row)
 [0.707  0      0.707]            [0.401 0.198 0.401]   sum = 1
 [0      0.707  0.707]     ->     [0.198 0.401 0.401]   sum = 1
 [0.707  0.707  1.414]            [0.248 0.248 0.503]   sum = 1

 output row 1 = 0.401*[1,2] + 0.198*[3,4] + 0.401*[4,6] = [2.599, 4.000]
```
Reading: token 1 attends 40% to itself, 20% to token 2, 40% to token 3 (it shares a feature with 3). Its new vector is a context-weighted blend.
Lecture slide 14: the same word **"bank"** gets different weights over "river/erodes" vs "deposit/money", so it gets a **context-dependent meaning**.

### 1.3 The rest of the architecture (paper)
- **Multi-head attention:** run h attention operations in parallel with different learned projections, then concatenate and project. Paper: **h = 8**, d_k = d_v = d_model/h = **64**,
  d_model = **512**. Because each head is small, total cost ≈ one full-size head. Different heads learn different relations (lecture slide 18: dog–cat, dog–bark, dog–park).
- **Positional encoding:** with no recurrence the model has no sense of order, so a position signal is **added** to the embeddings. Paper: **sinusoidal** functions of different
  wavelengths (learned position embeddings gave nearly identical results).
- **Feed-forward sub-layer** applied to each position (d_model 512 → d_ff 2048 → 512).
- **Residual connection + layer normalisation** around every sub-layer.
- **Encoder:** stack of N = **6** identical layers (self-attention + FFN). **Decoder:** N = **6** layers with an extra attention over the encoder output.
- **Masking in the decoder:** future positions are set to −∞ before softmax so a token cannot look at later tokens; this keeps generation **auto-regressive** (one token at a time).
- Three uses of attention: encoder self-attention; masked decoder self-attention; encoder–decoder ("cross") attention.
- **Modern LLMs** (GPT, Llama, Gemma) keep the **decoder-style** stack and are trained on **next-token prediction**. Any text is training data because the "label" is just the next token
  (lecture slides 11–12, 25: frontier models see 10–35 trillion tokens).

### 1.4 Why divide by √d_k?
If q and k have independent components with mean 0 and variance 1, then q·k has **variance d_k** (paper footnote). Large scores push softmax into regions with tiny gradients.
Check (simulation of 20,000 random pairs): standard deviation of q·k was **2.0** for d_k = 4, **8.0** for d_k = 64, **22.5** for d_k = 512 (= √d_k each time). Dividing by √d_k restores unit scale.

### 1.5 RNN vs Transformer (paper Table 1)

| | RNN | Self-attention |
|---|---|---|
| **Sequential operations** | **O(n)**: token t waits for t−1 | **O(1)**: all positions at once |
| Parallel training over a sentence | No | **Yes**: big matrix multiplications that GPUs are built for |
| **Maximum path length** between any two tokens | **O(n)** | **O(1)** |
| Long-range dependency | Info must pass through up to n steps (vanishing gradients) | Direct link in one layer |
| Complexity per layer | O(n·d²) | O(n²·d) |

**Efficiency evidence (paper):** 28.4 BLEU on WMT14 English→German; 41.8 BLEU on English→French after **3.5 days on eight GPUs**, a small fraction of earlier training cost.
**Why the extra scale was possible:** because training is parallel, you can train on far more data in the same time; this is why LLMs became feasible.

**The honest caveat (earns critical-evaluation marks):** per layer, attention does *more* arithmetic when sequences are long. Using n = 1000 tokens, d = 512:
attention n²·d ≈ **512 M** operations vs RNN n·d² ≈ **262 M**. Attention is cheaper only when **n < d**. Its advantages are **parallelism and path length**, and its cost is
**memory/compute growing with n²**, which limits context length.

### Mark scheme (what to include)
- [ ] Attention only; no recurrence/convolution.  - [ ] Q, K, V and the softmax(QKᵀ/√d_k)V formula.  - [ ] Multi-head attention.
- [ ] Positional encoding is needed.  - [ ] RNN is sequential (O(n)); Transformer parallel (O(1) sequential steps).  - [ ] Shorter paths → long-range dependencies.
- [ ] A trade-off (O(n²)).
**Common mistakes:** saying the Transformer is "faster because it does less work" (it is faster because it is parallel); forgetting positional encoding; mixing up encoder-only/decoder-only.

---

## Q2. The logit lens and the three stages of next-token processing

### 2.1 What a Transformer does layer by layer
Each layer **adds** an update (attention + FFN) to a running vector for each token, the **residual stream**. Only after the last layer is the stream converted into word probabilities by the
output matrix (**unembedding / `lm_head`**) and **softmax**.

### 2.2 The logit lens (nostalgebraist; lecture slide 21)
**Idea:** apply that same final read-out to the residual stream **after every layer**, as if the model had to answer *now*. You see what the model "believes" at layer 1, 2, ... N.
In the lab:
```
 for each layer i:
     act_last = hidden_states[i][0, -1, :]      # state of the LAST input token
     logits   = model.lm_head(act_last)         # same output layer as the final layer
     probs    = softmax(logits)
     show top-5 tokens and their probabilities
```
**What it shows (slide 21):** guesses go **from noisy to accurate**; the **input token is rapidly scrambled** (the model does not keep the raw input in the stream); it makes the internal prediction
process visible. **Caveat:** middle layers were never trained to be read by the final layer, so it is a rough lens.

### 2.3 The three stages (Merullo 2023, slide 22)
The paper treats a fact as a function: `y = get_capital(x)`, with the **argument x** (Poland) and the **output y** (Warsaw). The token passes through:

| Stage | What the logit lens shows | Mechanism |
|---|---|---|
| **A. Argument formation** | The top token becomes the **subject** being asked about (x) in earlier-to-middle layers | Earlier layers (attention gathers context) prepare x |
| **B. Function application** | The top token **abruptly switches from x to y** (an "X-shaped" crossing when plotting the ranks of x and y) | An **FFN layer** adds an update vector **o** to the residual stream (`x_{i+1} = x_i + o_i`), typically about **2/3 of the way** through the network |
| **C. Saturation** | The answer is already there, and the following layers **stop changing it** (Geva et al.'s "saturation events") | Remaining layers just reinforce/preserve |

**Vector arithmetic (the paper's title):** the FFN output behaves like a **word2vec-style offset**: `Poland + o_capital ≈ Warsaw`. The same vector patched into the context of "China" produces
**Beijing** (Poland:Warsaw :: China:Beijing). The mechanism appears in nine tasks (capitals, uppercasing, past tense...), becomes **sharper in bigger models**, and happens at similar *relative* depths.

### 2.4 Your lab run (Llama-3.2-1B-Instruct, 16 layers, "The capital of England is")

| Layer | Top-1 token (prob) | Meaning |
|---|---|---|
| 1 | is (0.00001) | copies the last input token |
| 2 | is | copy |
| 3 | was | changes tense |
| 4 | capital | pulls a word from the prompt |
| 5 | currency | wrong association |
| 6 | capitals | |
| 7 | ptive | noise |
| 8 | stad | noise (my guess: "stad" means "city" in some languages) |
| 9 | olit | noise (my guess: a word fragment, possibly heading towards a city name; not verified) |
| **10** | olit (0.00002); **"Lond", "London" are now in the top 3** | function being applied |
| **11** | **London** (0.00002) | **function application complete**: 11/16 = **69%** depth ≈ "2/3" |
| 12 | London (0.00005) | saturation begins |
| 13 | London (0.00007) | |
| 14 | London (0.00008) | |
| 15 | London (0.00017) | |
| **16** | **London (0.62351)**; "not" 0.094, "a" 0.044 | final distribution |

**How to read it:** layers 1–3 = still near the raw input; 4–9 = searching (words about capitals, then fragments); 10–11 = answer found; 12–16 = saturation, with confidence finally becoming high at the last layer.
**Honest caveats (say these):**
1. Probabilities of 0.00001–0.0001 are near uniform over a vocabulary of about 128,000 tokens, so early-layer readings are weak; only layer 16 is confident.
2. In this 1B model the **argument "England" is not in the top-5 before function application**, so Argument Formation is not cleanly visible. Merullo reports the pattern sharpens with model size.
3. The lab's prompt options (Tesla CEO, first US president, Brunel VC) are for you to try; the Brunel VC one is a good **hallucination test** (small models often get names wrong).

### Mark scheme
- [ ] Logit lens = decode the hidden state of every layer with the output layer.  - [ ] noisy → accurate; input scrambled.
- [ ] Name and explain the three stages with the capital example.  - [ ] FFN update, ≈ 2/3 depth.  - [ ] Cite the lab: "London" appears around layer 10–11 of 16, probability 0.62 at the end.
**Common mistakes:** thinking the model "knows" the answer from layer 1; saying attention applies the function (the paper finds mostly **FFN**); forgetting saturation.

---

## Q3. Principles for crafting effective instructions and prompts

### 3.1 Why prompts matter
LLMs predict the next token given the text so far, so **the wording of the prompt is the only control** you have at inference time. Lecture: *"Tell me about elephants."* (vague, unbounded) vs
*"Give three facts about elephant social behaviour."* (specific, bounded). Changing the wording changes the answer.

### 3.2 Evidence (Bsharat 2023)
**26 guiding principles** in **5 categories**, tested on LLaMA-1/2 (7B, 13B, 70B) and GPT-3.5/4. On GPT-4 they raised response **quality by 57.7%** and **correctness by 36.4%** on average (paper);
larger models gain more. The paper's six design aims: **conciseness and clarity, contextual relevance, task alignment, example demonstrations, avoiding bias, incremental prompting**.

### 3.3 The 26 principles (by the paper's numbering and category)

| Category | # | Principle |
|---|---|---|
| **Prompt structure & clarity** | 2 | Include the **intended audience** ("the audience is an expert in the field") |
| | 4 | Use **affirmative directives** ("do ...") rather than "don't" |
| | 12 | Use leading words such as **"think step by step"** |
| | 20 | **Output primer**: end the prompt with the start of the expected answer |
| | 17 | Use **delimiters** |
| | 8 | Format as `###Instruction###`, then `###Example###` / `###Question###`, separated by line breaks |
| **Specificity & information** | 7 | **Few-shot** (example-driven) prompting |
| | 5 | "Explain X in simple terms / like I'm 11 / as a beginner in [field]" |
| | 13 | Ask for an **unbiased** answer without stereotypes |
| | 26 | "Use the same language as the provided paragraph" |
| | 24 | Give the start of a text and ask the model to continue it consistently |
| | 25 | State the **requirements** the output must obey (keywords, rules, hints) |
| | 15 | "Teach me X and include a test at the end; don't give the answers beforehand" |
| | 21 | "Write a detailed [essay] on [topic], adding all necessary information" |
| **User interaction & engagement** | 14 | Let the model **ask you questions** until it has enough information |
| **Content & language style** | 22 | To edit text, improve grammar/vocabulary but **keep the original style** |
| | 9 | Use "Your task is" and "You MUST" |
| | 10 | Use "You will be penalized" |
| | 16 | **Assign a role** to the model |
| | 11 | Use "Answer a question given in natural language form" |
| | 1 | **No need to be polite** ("please", "thank you") |
| | 18 | Repeat key words/phrases |
| | 6 | Offer a tip ("I'm going to tip $xxx for a better solution") |
| **Complex tasks & coding** | 3 | **Break a complex task into a sequence of simpler prompts** |
| | 23 | For multi-file code, ask for a script that creates/edits the files |
| | 19 | Combine **chain-of-thought with few-shot** |
(That is all 26; some, like the tip and "you will be penalized", are the paper's empirical findings and are not guaranteed to work on every model.)

### 3.4 System prompt vs user prompt (slide 37)
- **System prompt:** set by developers/the application. Controls overall **behaviour, tone, persona, limits**. Example: *"You are a helpful assistant."* Used in Lab 9.2.
- **User prompt:** typed by the user. Asks for the **specific output**. Example: *"List three benefits of exercise."*
(Slide 38 shows a rumoured ChatGPT system prompt, i.e. real products are steered by long hidden system prompts.)

### 3.5 Worked example: improving a prompt
Bad: *"Summarise this."* → unknown length, format, audience.
Better (principles 2, 16, 25, 17, 20):
```
 You are a financial journalist.                      (role)
 Summarise the article between ### markers for a busy executive (audience),
 in exactly three bullet points, each under 20 words.  (requirements)
 ###
 {article text}
 ###
 Summary:                                             (output primer)
```
This matches your Lab 9.1 prompts: "exactly three bullet points" and "no more than 50 words for a busy executive".

### Mark scheme
- [ ] Prompt wording changes the output; give a poor/good pair.  - [ ] Clear, specific, focused.  - [ ] Role, audience, constraints, examples (few-shot), step-by-step, delimiters, decomposing tasks.
- [ ] System vs user prompt.  - [ ] Cite Bsharat (26 principles; +57.7% quality on GPT-4).
**Common mistakes:** listing generic "be clear" with no concrete principles; not distinguishing system from user prompt.

---

## Q4. Low-Rank Adaptation (LoRA) and its parameter efficiency

### 4.1 Context
**Pre-training** teaches next-token prediction, not specific tasks. **Instruction fine-tuning** (slides 40–41) continues training on a small labelled set, starting from parameters "close to pre-train".
Full fine-tuning updates **all** parameters. For GPT-3 175B that means a separate 175B-parameter copy and huge optimiser memory for every task.

### 4.2 Mechanism (Hu 2021)
1. Take a pre-trained weight matrix **W₀** of size d × k. **Freeze it** (no gradient updates).
2. Represent the update as a **low-rank product**: `W = W₀ + ΔW = W₀ + B·A`, with **B: d × r**, **A: r × k**, and **rank r ≪ min(d, k)**.
3. Forward pass: **`h = W₀x + B·A·x`**, with ΔW·x scaled by **α/r** (α is just set to the first r tried, so it need not be re-tuned).
4. **Initialisation:** A ~ random Gaussian, **B = 0** ⇒ ΔW = 0 at the start, so fine-tuning begins exactly at the pre-trained model.
5. Train **only A and B** with the usual loss (next token on your small labelled data).
6. Applied mainly to the attention matrices **W_q and W_v** in each layer.
7. **Deployment:** compute W = W₀ + B·A once and **merge** it, so inference has **no extra latency**.

**Why does a low rank suffice?** Hypothesis from earlier work: when adapting to a task, pre-trained models have a low **"intrinsic dimension"**, so the weight *change* has low "intrinsic rank". Even r = 1–4 can work
(paper: trainable parameters can be **0.01%** of the full model).

### 4.3 Parameter arithmetic (hand-check)
```
 One matrix, d = k = 4096, rank r = 8:
   full fine-tune of that matrix : d * k          = 4096 * 4096 = 16,777,216
   LoRA (A and B)                : r*(d + k)      = 8 * 8192    =     65,536
   fraction = 65,536 / 16,777,216 = 0.39%   -> 256 times fewer

 Tiny picture: d = k = 4, r = 1:   W: 16 numbers;  B (4x1) + A (1x4) = 8 numbers.
 Rule: cost = r(d+k), not d*k; doubling r doubles LoRA cost; width d enters linearly instead of quadratically.

 GPT-3 175B (96 layers, d_model = 12288), r = 4 on W_q and W_v only:
   number of adapted matrices = 96 * 2 = 192
   params = 2 * 192 * 12288 * 4 = 18.9 million   (paper: "18M", about 35 MB)
```

### 4.4 Paper's results (GPT-3 175B)
- Trainable parameters reduced by **10,000×**; GPU memory by **3×** (training VRAM **1.2 TB → 350 GB**, because optimiser states are not stored for frozen weights).
- Checkpoint size **350 GB → 35 MB** (r = 4, W_q and W_v).
- Quality **on par with or better than** full fine-tuning on RoBERTa, DeBERTa, GPT-2 and GPT-3.
- **No additional inference latency** (unlike adapter layers, which add sequential layers and slow inference noticeably at small batch sizes).
- **Task switching** is cheap: keep one frozen base model, swap tiny LoRA files.

### 4.5 Limitations (for "evaluate")
Needs the base weights (so open models such as Llama, lecture slides 47–49: "everything changed with Llama"); rank and which matrices to adapt must be chosen; if the task needs changes that are
truly high-rank, a low r may underfit; still needs labelled data.

### Mark scheme
- [ ] Freeze W₀.  - [ ] ΔW = B·A with rank r ≪ d,k.  - [ ] Only A, B trained.  - [ ] Parameter count r(d+k) vs d·k with a number.  - [ ] B = 0 at init.  - [ ] Merge → no inference latency.
- [ ] Memory/storage savings (10,000×, 3×).
**Common mistakes:** saying LoRA compresses the model; saying it adds layers (that is adapters); forgetting that W₀ stays unchanged.

---

## Q5. RLHF: the reward model and the importance of the KL constraint

### 5.1 Why RLHF exists
Instruction fine-tuning needs a human-written **full ideal answer** for each prompt (hard, slow). For many goals (helpful, polite, safe, better summary) it is much easier to **look at several answers and say which is better**.
Lecture slide 54: fine-tuning data = full output, difficulty hard, process = adaptor; **RLHF data = only a choice**, difficulty easy, process = **reward model**. Ziegler found humans were inconsistent giving absolute scores,
while **comparisons were almost as informative**.
Reinforcement learning differs from supervised/self-supervised learning (slide 51): there is no "correct output" label, only a **reward signal** for the model's own generated output.

### 5.2 The pipeline (Ziegler 2020)
1. **Pre-train** language model **ρ** (paper: 774M-parameter GPT-2).
2. **Collect human feedback:** for a prompt x, sample **four** responses y₀..y₃ from ρ; a human labeller picks the best (index b).
3. **Train the reward model** r(x, y) (initialised from ρ, with a new linear output layer) using
```
 loss(r) = − E[ log( exp r(x, y_b) / Σ_i exp r(x, y_i) ) ]
```
   i.e. a softmax over the four responses that rewards giving the human's choice the highest score.
4. **Optimise the policy π** (initialised as a copy of ρ) with RL (**PPO**) to maximise the **modified reward** `R(x,y) = r(x,y) − β·log(π(y|x)/ρ(y|x))`.
5. **Optionally repeat:** collect new human labels on π's outputs and retrain r ("online" version).

### 5.3 The reward model
- A neural network that **predicts how good a response is** from human rankings (slide 52).
- It is a **proxy for the human**: after training, it can score millions of responses automatically, which RL needs.
- **Limits:** it only approximates human judgement and is valid only for the kind of text it was trained on. (Lecture: reward models are not open; only the big app owners have them.)

**Worked example of the loss:** suppose the model scores four answers r = [1.2, 0.3, −0.5, 0.0] and the human chose answer 0. softmax gives P(chosen) = **0.529**, loss = −ln 0.529 = **0.637**.
If training pushes r(y₀) up, P(chosen) → 1 and loss → 0.

### 5.4 The KL constraint
**KL divergence** measures how different the tuned policy π is from the original ρ (0 when identical). The penalty term `β·log(π/ρ)` has expectation **β·KL(π, ρ)**. Why it is essential:
1. **Prevents reward hacking / model drift (slide 53):** RL will find outputs the reward model likes but humans would not (gibberish with the right keywords). Without KL, π can wander anywhere.
2. **Keeps π in the region where r is reliable:** r was trained on samples from ρ, so its scores are meaningless far away.
3. **Retains general ability and fluency (slide 53: quality, stability):** the model must stay coherent while adopting the new preference. In Ziegler's style tasks humans judged *style* and the KL term kept text **coherent and on topic**.
4. Acts like an **entropy bonus**, discouraging collapse onto one repeated output.
β sets the trade-off: **small β = more reward, more risk; large β = safe, little change.**

**Worked example (β = 0.1, r = 2):**
| π(y)/ρ(y) | β·ln(ratio) | R |
|---|---|---|
| 1 | 0.00 | **2.00** |
| 9 | 0.22 | **1.78** |
| 99 | 0.46 | **1.54** |
The further the policy moves from the original, the more the reward is discounted. A tiny numeric KL illustration: p = (0.7, 0.2, 0.1) vs q = (0.4, 0.4, 0.2) gives KL(p‖q) = **0.184 nats**; identical distributions give 0.

### 5.5 Results and a warning (paper)
About **5,000** human comparisons were enough for stylistic continuation (positive sentiment, physical descriptions); **60,000** for summarisation. The summariser ended up **copying whole sentences** from the input, which
labellers rated highly because they used copying as a quick accuracy check: a real example of optimising a proxy.

### Mark scheme
- [ ] Why preferences instead of written answers.  - [ ] Human ranks outputs → **reward model** trained.  - [ ] RL (PPO) tunes the LLM to maximise reward.
- [ ] **KL penalty** vs original model; list ≥ 2 reasons (drift/reward hacking; reward model validity; retain fluency).  - [ ] Role of β.  - [ ] Mention limitation (reward model is a proxy).
**Common mistakes:** saying the KL term is "to make the model more accurate"; confusing the reward model with the LLM; claiming the human scores every output during RL training.

---

## Q6. Abstractive summaries from long texts with a pre-trained LLM (Gemini API)

### 6.1 Abstractive vs extractive
**Extractive** (older, rule-based): select the most important sentences verbatim. **Abstractive** (LLMs): write **new sentences** that paraphrase and compress the meaning. Benefit: fluent and concise. Risk: **hallucination**
(adding or distorting facts), so it must be checked.

### 6.2 Procedure (Lab 9.1)
1. **API key:** create in Google AI Studio; keep it in a secret store (Colab Secrets `GEMINI_API_KEY`), **not** in code.
2. **Client:** `from google import genai; client = genai.Client(api_key=...)`.
3. **Model:** e.g. `gemini-2.5-flash` (fast, cheap) or `gemini-2.5-pro` (stronger reasoning).
4. **Prompt structure:** `instruction + constraints (length, format, audience) + delimiter + the text`.
   `prompt = f"Summarise the following text in plain English:\n\n{input_text}"`
5. **Call:** `response = client.models.generate_content(model=model_name, contents=prompt)`; `summary = response.text.strip()`.
6. **Vary the prompt** (3 prompts in the lab), **evaluate** (accuracy, coherence, length control), **repeat with another text**.
7. **For very long inputs** exceeding the context window: split into chunks, summarise each, then summarise the summaries (a "map-reduce" approach).

### 6.3 Your run (BBC: IFS says households face a "dismal" rise in disposable income)

| Prompt | Words | What happened |
|---|---|---|
| "Summarise the following text in plain English" | **192** | 4 paragraphs: 0.5% annual growth over five years vs more than 2% historically; the IFS claim the government broke its tax promise (income tax threshold freeze, pension cap); Starmer/Reeves deny it; the Budget did little for growth |
| "exactly three bullet points" | **117** | Exactly 3 bullets (but a filler opening line "Here's a summary..."), each rich in detail |
| "no more than 50 words for a busy executive" | **56** | **Over the limit by 6 words**; content compressed to the key message |
- The source contained **cookie banners and menus**; none appeared in the summaries: the model picks out the relevant content.
- **Findings:** *format* constraints (3 bullets) are followed reliably; *length* constraints only approximately; shorter summaries drop detail (the 50-word one drops all names and nearly all policy detail).

**Accuracy check against the article text in your notebook (I read the source and compared):**
| Summary claim | Source says | Verdict |
|---|---|---|
| IFS: 0.5% annual growth over five years, versus over 2% historically | Yes (OBR forecast; Miller: over 2% a year mid-1980s to mid-2000s) | ✔ |
| Extended income-tax threshold freeze and a pension contribution cap | Freeze extended three years beyond 2028; £2,000 cap on pension salary sacrifice | ✔ |
| 192-word: "**They** [the IFS] predict ... 0.5%" | The forecast is the **OBR's**; the IFS analysed it | ✘ small **attribution slip** |
| 192-word: "Both [Starmer and Reeves] state that some policies will 'impact working people'" | Only **Reeves** said that; Starmer said the contribution was "fair and necessary" | ✘ blurs who said what |
| 3-bullet: Reeves defended choices "to address **fiscal pressures**, cut NHS waiting lists ..." | Reeves cited NHS waiting lists, child poverty, cost of living; "fiscal pressures" is the IFS's wording | ✘ mild embellishment (and "lift children out of poverty" is missing) |
| 50-word: "2025 Budget ... effectively breaching tax pledges" | Miller: "I would call that a breach of the manifesto" | ✔ (but stated as fact, not as the IFS's view) |
| Key figure "nearly £104 a year" | In the article | not in any summary (omitted) |
So the summaries are **mostly accurate but not error-free**: the errors are about **who said what** and small paraphrase drift, exactly the kind of mistake a quick read misses. This is the point for the "evaluate accuracy" requirement.

### 6.4 Evaluation rubric for the lab report

| Criterion | What to check | Result in this run |
|---|---|---|
| **Accuracy** | Every claim and number is in the source; nothing invented; attribution right (IFS vs government) | Main claims match the article's themes; verify numbers manually |
| **Coherence** | Reads as connected text, correct pronouns/order | Good in all three |
| **Length control** | Words vs instruction | 3 bullets ✔, 50 words ✘ (56) |
| **Coverage vs brevity** | What was lost | Detail drops as the limit tightens |
| **Prompt sensitivity** | Same text, different instruction → different structure/tone | Clear |

Possible extra: automatic metrics such as ROUGE exist (not taught in this module), but for abstractive summaries human or LLM judging is more reliable.

### Mark scheme
- [ ] Abstractive vs extractive.  - [ ] API key, client, model, prompt structure, `generate_content`.  - [ ] Prompt design (length, format, audience).  - [ ] Evaluation criteria.  - [ ] Hallucination risk; chunking for long texts.
**Common mistakes:** saying the model "copies" sentences (that is extractive); no mention of how to evaluate.

---

## Q7. Building a multi-turn chatbot: steps and the core technical challenge

### 7.1 The key fact: LLMs are stateless
An LLM maps an input token sequence to a next-token distribution. It does **not** remember the last conversation. Memory exists only because the application **re-sends the whole conversation each turn** (lab text).
Without the history, "What MSc programmes do **they** offer?" has no referent.

### 7.2 Steps (Lab 9.2)
1. **Choose the model:** a local open model, e.g. `google/gemma-3-1b-it` (1B parameters, instruction-tuned), via Hugging Face `transformers` (needs an access token for gated models), **or** an API.
2. **Tokenizer + model:** `AutoTokenizer` (text ↔ token ids) and `AutoModelForCausalLM` (the next-token predictor); `pipeline("text-generation", ...)`.
3. **System prompt:** `system_prompt = "You are a helpful assistant. Please reply clearly and briefly."` sets tone/persona/limits.
4. **History list:** `messages = [{"role":"system","content":system_prompt}]`.
5. **Loop** until the user types `quit`:
```
 text = input("You: ")
 messages.append({"role":"user","content":text})              # add the user turn
 response = pipe(messages, max_new_tokens=256)                 # whole history in
 bot_reply = response[0]["generated_text"][-1]["content"]      # last message = new reply
 messages.append({"role":"assistant","content":bot_reply})     # store it so next turn sees it
 print("Bot:", bot_reply)
```
6. The **chat template** converts the role-tagged list into the exact token format the model was fine-tuned on. (The lab's first version, `get_llm_reply(prompt)`, took only a single prompt: it had **no memory**.)
7. Control generation (`max_new_tokens`), test, change the system prompt, and optionally swap to an API model.

### 7.3 The core technical challenge: managing context/memory
- **Why history is needed:** reference resolution, consistency, following earlier instructions.
- **The cost grows every turn:** every request carries all earlier turns. If each exchange is about 200 tokens, turn t sends about 200·t tokens, so after 20 turns one request is ~4,000 tokens and the **total processed over 20 turns is 200·(1+2+...+20) = 42,000 tokens**
  versus 4,000 if the model were stateful. Growth is quadratic in the number of turns: higher **latency and cost**.
- **Hard limit:** the **context window**. When history exceeds it, something must go. Strategies: **truncate** old turns (keep system prompt + last N), **summarise** earlier turns into a short memory, or **retrieve** only relevant earlier parts.
- Related challenges: keeping the **system prompt/persona** in force over long chats; **hallucination**; safety.

### 7.4 Your lab run (Gemma-3-1B-it)
- Turn 1: "how much do you know about Brunel University?" → fluent but contains **errors**: "situated in Bath, Somerset, close to Stonehenge" (**false**: Brunel University London is in Uxbridge, west London) and "founded in 1876".
  A 1B-parameter model has weak **world knowledge** (lecture slide 8: linguistic knowledge is easy, world knowledge is much harder).
- Turn 2: "do they have a computer science department?" → answered **about Brunel** although "Brunel" was not repeated: **history worked**.
- Turn 3: "what Msc programmes do they offer?" → again resolved "they"; listed programmes and gave a URL it could not have verified.
- The reply was formatted with bullet lists, bold and a closing offer; the system prompt asked for "briefly", so you can test how changing it changes length/tone (lab requirement 2).

### 7.5 Trade-offs: small local model vs larger/API model (lab requirement 4)

| | Small (≈1B) local | Large / API (e.g. Gemini) |
|---|---|---|
| **Speed** | Fast on a small GPU/CPU, no network | Network latency, rate limits |
| **Cost / privacy** | Free to run, data stays on your machine | Paid/quota; data sent to provider |
| **Quality & depth** | Shallow, **more hallucination**, weaker reasoning, short context | Much better knowledge, reasoning, longer context |
| **Control** | Can be fine-tuned (e.g. LoRA) | Limited to prompting/fine-tuning offered |
Rule: choose small for prototypes, offline or private use; larger for answers that must be correct.

### Mark scheme
- [ ] LLM stateless; history re-sent each turn.  - [ ] Roles: system/user/assistant list; loop and append both sides.  - [ ] Tokenizer/model/pipeline; system prompt.
- [ ] Core challenge = **context window / memory management** (truncate, summarise, retrieve) with cost growth.  - [ ] Hallucination/quality trade-off for small models.
**Common mistakes:** saying the model "remembers" between calls; giving only the code loop without naming the challenge.

---

## What is not done yet
- `Week9_DETAILED_NOTES.md` (full lecture notes: history N-gram→CNN→RNN→Transformer, model sizes, unified multi-task models, instruction fine-tuning vs RLHF, Llama, etc.) and the Lab 9 write-ups.
- `Week9_QUIZ_with_answers.md`: **paste the Week 9 quiz** and I'll answer it in the usual format (score key, most-missed line, grouped by sub-topic).
