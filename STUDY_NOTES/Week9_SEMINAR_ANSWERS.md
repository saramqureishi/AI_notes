# CS5707 — WEEK 9: LARGE LANGUAGE MODELS — Seminar Answers

> Sources: Lecture W9 (all 55 slides, text only; figure slides have no text), Lab W9 + the three notebooks
> (outputs are from Sara's own Colab runs), and the papers: Vaswani 2017, Merullo 2023, Bsharat 2023,
> Hu 2021 (LoRA), Ziegler 2020. Numbers marked "paper" were checked against the PDF text.
>
> **Honest gaps**
> - I did NOT re-run the labs. They need a Gemini API key and gated models (Gemma, Llama) that I can't access.
>   Lab results below are quoted from the outputs saved in your notebooks.
> - The full Week 9 detailed notes and the quiz are not written yet (see end).
> - Exam style: these are "describe / explain / why" questions. Learn the **bold** points; they are the mark-earners.

---

## Q1. Transformer: attention only; why better than RNNs for parallelism and efficiency

**Core idea (Vaswani 2017):** a sequence model built **solely on attention**, "dispensing with recurrence and convolutions".

**How self-attention works**
- Each token is turned into three vectors: **query (Q)**, **key (K)**, **value (V)**.
- Each token's query is compared with every token's key; the scores are turned into weights by softmax; the output is the weighted sum of values:
```
 Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V
```
- Result: an **attention matrix** where row i says how much token i looks at every other token (lecture slides 16–17; each row sums to 1).
  Slide 14: "bank" attends to "river/erodes" in one sentence and to "deposit/money" in another, so the **same word gets a context-dependent meaning**.
- **Multi-head attention** (paper: h = 8 heads, d_k = 64, d_model = 512): several attention patterns in parallel, each capturing a different
  relationship (lecture: dog–cat, dog–bark, dog–park).
- **Positional encoding** is added because with no recurrence the model otherwise has no idea of word order.
- Blocks are stacked (many layers): attention + feed-forward network, with residual connections and layer norm. Paper: 6 encoder + 6 decoder layers.
- LLMs use the decoder side to **predict the next token**; this needs no labels, so any text is training data (slides 11–12, 25).

**Why better than RNNs (the comparison the question wants)**

| | RNN | Transformer self-attention |
|---|---|---|
| Sequential operations per layer | **O(n)**: token t needs the hidden state of t−1 | **O(1)**: all positions computed at once |
| Parallelism in training | **Cannot parallelise within a sentence** | **Fully parallel** over all positions: ideal for GPUs |
| Max path length between two tokens | **O(n)** | **O(1)**: any token can look at any other directly |
| Long-range dependencies | Information must survive many steps (vanishing gradient) | One step |
| Cost per layer | O(n·d²) | O(n²·d) |
(Numbers from Table 1 of the paper.)

- **Efficiency:** the paper reached **28.4 BLEU** (EN→DE) and **41.8 BLEU** (EN→FR) after **3.5 days on eight GPUs**, a small fraction of earlier training cost.
- **Honest trade-off (earns "critically evaluate" marks):** self-attention costs O(n²·d), which is cheaper than an RNN only when sequence length n < d.
  For n = 1000, d = 512: attention ≈ 512 M operations vs RNN ≈ 262 M (my calculation). So the win is **parallelism and path length**, not fewer
  operations; very long inputs make memory grow quadratically.

---

## Q2. The logit lens and the three processing stages

**Logit lens** (nostalgebraist; lecture slide 21):
- Normally the model only reads out a prediction at the **last** layer.
- The logit lens takes the **hidden state after every layer** and pushes it through the **final output layer (`lm_head`, the unembedding) + softmax**,
  as if the model had to answer *right now*. This gives "what the model would guess at layer k".
- Findings: guesses move **from noisy to accurate** across layers; the input token is **rapidly scrambled** and not kept as is; it shows the internal prediction process.
- Lab code: `logits = model.lm_head(hidden_states[k][0,-1,:]); probs = softmax(logits); topk(5)`, printed for each of the 16 layers.

**The three stages (Merullo 2023, slide 22)**. The model treats a fact as a function `y = get_capital(x)`:

| Stage | What happens | Example (capital of Poland) |
|---|---|---|
| **A. Argument Formation** | Early/middle layers surface the **argument x**; the top token is the subject the question is about | "Poland" appears |
| **B. Function Application** | The representation **abruptly switches** from x to the function output y. Done by a **feed-forward (FFN) layer** adding an update vector o to the residual stream, at about **2/3 of the way** through the network | Poland → "Warsaw" (`capital_of`) |
| **C. Saturation** | The model has found the answer and **stops changing it** in the remaining layers | "Warsaw" stays on top |

Extra (paper): the FFN update acts like **word2vec-style vector arithmetic**: `Poland + o ≈ Warsaw`, and the same `o` patched into another context turns China into Beijing
(Poland:Warsaw :: China:Beijing). The pattern becomes **sharper in larger models** and happens at similar *relative* depths.

**Lab evidence** (Llama-3.2-1B-Instruct, 16 layers, prompt "The capital of England is"):

| Layers | Top token(s) | Interpretation |
|---|---|---|
| 1–3 | is, was | just copies / re-tenses the last input token |
| 4–6 | capital, currency, capitals | stuck on prompt words, wrong direction (lab text: "confused") |
| 7–9 | stad, dém, olit, ... | noise |
| **10–11** | **London appears** (layer 10 as "Lond/London", top at 11) | **function application** (layer 11 of 16 = 69% depth, matches "~2/3") |
| 12–16 | London | **saturation**: London stays on top; probability climbs 0.00005 → 0.00017 (layer 15) → **0.62351 at layer 16** |

**Caveats (say these in an exam, it shows understanding):**
- Intermediate probabilities are around 0.00001, close to uniform over a ~128k-token vocabulary, so early-layer readings are weak signals and only layer 16 is confident.
- In this 1B model the **argument "England" is not visible in the top-5 early on**; Argument Formation is cleaner in larger models (Merullo).
- The logit lens is an approximation: middle layers were not trained to be read by the final output layer.

---

## Q3. Principles for crafting effective instructions and prompts

**Why it matters:** wording changes the answer. Lecture example: poor "Tell me about elephants." vs good **"Give three facts about elephant social behaviour."**
**Evidence (Bsharat 2023):** **26 principles** in **five categories**; tested on LLaMA-1/2 (7B, 13B, 70B) and GPT-3.5/4. On GPT-4 they raised response
**quality by 57.7% and correctness by 36.4% on average** (paper); gains are bigger for larger models.

| Category | Principles (examples) |
|---|---|
| **Prompt structure & clarity** | State the **intended audience** ("expert in the field"); use **affirmative directives** ("do") not "don't"; "**think step by step**"; **output primer** (end the prompt with the start of the answer); **delimiters** (`###Instruction###`, `###Example###`, line breaks) |
| **Specificity & information** | **Few-shot examples**; explain-like-I'm-11 style requests; explicit requirements/keywords the output must obey; "use the same language as the sample" |
| **User interaction** | Let the model **ask you questions** until it has enough information |
| **Content & language style** | **Assign a role** ("You are..."); "Your task is / You MUST"; **no need to be polite** (skip "please", "thank you"); to edit text, say to keep the original style |
| **Complex tasks & coding** | **Break complex tasks into simpler prompts**; combine **chain-of-thought with few-shot** |

**System vs user prompt (slide 37):** the **system prompt** is set by developers and shapes overall behaviour, persona and tone ("You are a helpful assistant");
the **user prompt** asks for the specific output ("List three benefits of exercise").
**One-line recipe:** clear task + audience/role + constraints (length, format) + examples + delimited input + ask for steps for hard problems.

---

## Q4. LoRA and why it is parameter-efficient

**Problem:** full fine-tuning updates **every** weight; for GPT-3 (175B) that means a 175B-parameter copy per task.

**Mechanism (Hu 2021):**
- **Freeze** the pre-trained weights W₀ (size d × k).
- Learn the update as a **product of two small matrices**: `ΔW = B·A`, with **B: d × r**, **A: r × k**, and rank **r ≪ min(d, k)**.
- Forward pass: **`h = W₀x + B·A·x`** (scaled by α/r). Only A and B get gradients.
- **Init:** A random Gaussian, **B = 0**, so ΔW = 0 at the start and training begins exactly at the pre-trained model.
- **Why low rank works:** hypothesis that weight *updates* during adaptation have a low "intrinsic rank", so a tiny subspace is enough.
- Usually applied to the attention matrices **W_q and W_v**.

**Why that is parameter-efficient (hand calculation)**
```
 d = k = 4096, r = 8
 full update of one matrix : d x k        = 16,777,216 parameters
 LoRA update               : r(d + k) = 8 x 8192 = 65,536 parameters
 ratio                     : 0.39%  (256 times fewer)
```
Cost grows with r(d+k) instead of d·k.

**Paper's headline numbers (GPT-3 175B):** **10,000× fewer trainable parameters**, **3× less GPU memory** (VRAM 1.2 TB → 350 GB, no optimiser state for frozen weights),
checkpoint **350 GB → 35 MB** (r = 4 on W_q, W_v, about 18 M parameters). Performance **on par or better than full fine-tuning**.

**Other advantages:** after training, **B·A is merged into W₀, so there is no extra inference latency** (adapter layers do add latency);
you can **swap small LoRA files per task** on one base model; fewer GPUs needed.
**Limits:** needs the pre-trained weights (open models such as Llama, lecture slides 47–49); r must be chosen.

---

## Q5. RLHF, the reward model and the KL constraint

**Why:** instruction fine-tuning needs humans to **write full ideal outputs** (hard). It is much easier for people to **choose** which of several outputs is better
(lecture slide 54: fine-tuning needs the full output, RLHF needs only a choice). Ziegler: asking for absolute scores was inconsistent; comparisons were almost as useful.

**Pipeline (Ziegler 2020):**
1. **Collect human preferences:** sample several responses (paper: 4 options) from the pre-trained model ρ; humans pick the best.
2. **Train the reward model r(x, y):** predicts how good a response is. It is initialised from the language model and trained with the loss
   `−log( e^{r(x,y_b)} / Σ_i e^{r(x,y_i)} )`, i.e. a softmax that should give the chosen response the highest score.
3. **Optimise the policy π** (the LLM being tuned) with **reinforcement learning (PPO)** to maximise the reward model's score.
4. (Optionally) keep collecting data and retrain r as π improves.

**Role of the reward model:** stands in for human judgement at scale: scores any new response automatically, so RL can run on millions of samples.
(Lecture: it is not open, only the big app owners have one.)

**The KL constraint:** the optimised reward is
```
 R(x,y) = r(x,y) − β · log( π(y|x) / ρ(y|x) )        (penalty = β · KL(π, ρ) on average)
```
KL divergence measures the **distance between the tuned model π and the original ρ**. It is needed to:
- **Stop model drift / "reward hacking":** otherwise π finds odd outputs the reward model scores highly but humans would not.
- **Stay where the reward model is valid:** r was trained on samples from ρ, so it is unreliable far away.
- **Keep fluency and general ability** (coherence, topicality) while adding the new preference. Slide 53: drift, stability, quality.
- Acts as an entropy bonus (keeps outputs varied).

**Small example (β = 0.1, reward r = 2):**
| π(y)/ρ(y) | penalty β·ln(ratio) | R |
|---|---|---|
| 1 (unchanged) | 0 | 2.00 |
| 9 | 0.22 | 1.78 |
| 99 | 0.46 | 1.54 |
The more the model moves from the original, the more reward it loses.

**Numbers (paper):** about **5,000** human comparisons were enough for stylistic text continuation; **60,000** for summarisation. The summariser learned to
**copy whole sentences** because labellers used copying as a quick accuracy check: a real example of exploiting the reward.

---

## Q6. Abstractive summaries with a pre-trained LLM (Gemini API)

**Abstractive vs extractive:** extractive picks sentences from the source; abstractive **writes new sentences** that capture the meaning.

**Steps (Lab 9.1):**
1. Get an API key (Google AI Studio) and store it as a secret (`GEMINI_API_KEY`); never hard-code it.
2. `from google import genai; client = genai.Client(api_key=...)`.
3. Choose a model, e.g. `gemini-2.5-flash` (fast, cheap) or `gemini-2.5-pro` (stronger).
4. Build the prompt = **instruction + constraints + the text**, e.g.
   `f"Summarise the following text in plain English:\n\n{input_text}"`.
5. `response = client.models.generate_content(model=model_name, contents=prompt)`; `summary = response.text.strip()`.
6. Try several prompts, then **evaluate**: **accuracy** (no invented or lost facts), **coherence**, **length control**.

**Your lab results** (BBC article on the IFS verdict on the Budget; model gemini-2.5-flash):

| Prompt | Output length | Observation |
|---|---|---|
| "Summarise in plain English" | 192 words | Four paragraphs; covers disposable income 0.5% a year, the broken-promise claim, the government's denial |
| "exactly three bullet points" | 117 words | Exactly 3 bullets, but opens with a filler line ("Here's a summary...") |
| "no more than 50 words for a busy executive" | **56 words** | **Limit exceeded by 6 words**: length control is approximate |
The pasted web page contained cookie-banner and menu text; the summaries ignored it, a sign the model separates relevant content from noise.
Lesson: **format constraints are followed well, word limits only roughly**; always verify facts against the source (hallucination risk) and
tell the model the audience (principle from Q3). For *long* texts that exceed the context window: split into chunks, summarise each, then summarise the summaries.

---

## Q7. A multi-turn chatbot: steps and the core challenge

**Key fact:** an LLM is **stateless**. It does not remember earlier turns. "Memory" exists only because the **whole conversation so far is sent back with every new request**.

**Steps (Lab 9.2):**
1. Load a model and tokenizer (e.g. `google/gemma-3-1b-it`, 1B parameters, via Hugging Face; needs an access token) or call an API.
2. Write a **system prompt** (persona, tone, limits), e.g. "You are a helpful assistant. Please reply clearly and briefly."
3. Keep a **history list** `messages = [{"role":"system",...}]`.
4. **Loop:** read user input → append `{"role":"user",...}` → pass the **entire list** to the model (the chat template turns it into tokens) →
   take the reply → append `{"role":"assistant",...}` → print → repeat until the user types "quit".

**Core technical challenge: context (memory) management.**
- The model must see the history to resolve references ("they", "that programme") and stay consistent.
- But the history **grows every turn**, and the model has a **fixed context window**; longer input = **more compute, latency and cost**.
  Solutions: truncate old turns, **summarise** older turns, keep only the system prompt + the latest N turns, or retrieve relevant parts.
- Related: keeping the **persona/instructions** stable over many turns, and **hallucination**.

**Lab evidence (Gemma-3-1B):** the follow-ups "do they have a computer science department?" and "what MSc programmes do they offer?" worked **because the history was passed back**
("they" = Brunel). But the first answer said Brunel is **"situated in Bath, Somerset, close to Stonehenge"**, which is wrong (Brunel University London is in Uxbridge, London),
and it gave a link it could not have checked. That is the small-model trade-off:

| | Small model (1B) | Large/API model |
|---|---|---|
| Speed / cost | fast, cheap, runs locally/offline | slower, paid or rate-limited |
| Privacy | data stays local | data sent to a provider |
| Quality / depth | **shallower, more hallucination** | better knowledge and reasoning |

---

## What is not done yet
- `Week9_DETAILED_NOTES.md` (full lecture notes), Lab 9 write-ups, and `Week9_QUIZ_with_answers.md`.
- **Paste the Week 9 quiz** and I'll answer it in the usual format (score key, most-missed line, grouped by sub-topic).
