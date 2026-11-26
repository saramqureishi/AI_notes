# CS5707 — WEEK 9 QUIZ (Large Language Models) — Answers

**Score key:** 1-B 2-B 3-C 4-C 5-C 6-C 7-A 8-B 9-C 10-B 11-C 12-B 13-D 14-B 15-B 16-C 17-C 18-B 19-B 20-C

**Most-missed (watch these):** **Q2** (the advantage is *constant sequential operations*, not hardware or memory claims), **Q12** (a word embedding is a *point in a vector space*; the "atomic/discrete" option describes one-hot encoding), **Q13** ("All of the above": don't stop at the first true option), **Q8** (the two stages are *pre-train, then adapt*; the RLHF stages are a different thing), **Q10 / Q20** (fine-tuning needs a *full output*; RLHF needs only a *choice*).

> Sources: Week 9 lecture slides, Vaswani 2017, Merullo 2023, Bsharat 2023, Hu 2021 (LoRA), Ziegler 2020, Russell & Norvig §25 (all in the repo), and the Lab 9 notebooks.
> "Slide n" = slide in `Lecture W9 Large Language Models.pdf`; "p. n" = PDF page. Several questions repeat in different wording (Q5, Q6/18, Q9/19, Q10/20, Q2/16): they were all answered the same way.
> Honest note: Q13 lists code generation and mathematical reasoning, which the lecture does not teach in detail; I answered it from the question's own wording ("All of the above" with three true statements) and from the lecture's summarisation and Q&A material.

---

## A. Transformer and attention

**Q1 → B. It relies solely on attention mechanisms, discarding recurrence and convolutions entirely.**
The Transformer paper's abstract describes a model "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely" (Vaswani p. 1).
*Trap A (RNN + CNN combination):* that is exactly what the Transformer removes. *Trap C (LSTMs):* an LSTM is a recurrent network, the thing being replaced. *Trap D (grammar/semantic parsing):* a rule-based NLP approach, not this architecture.

**Q2 → B. The Transformer connects all positions with a constant number of sequentially executed operations, allowing significantly more parallelization.**
Table 1 of the paper: self-attention needs O(1) sequential operations per layer, a recurrent layer needs O(n), so all positions can be computed at once (Vaswani p. 6; p. 2).
*Trap C (RNNs need more memory for weights):* the claim is about the order of computation, not weight storage. *Trap A (single-core CPUs):* RNNs can run on GPUs; they just cannot parallelise **across the positions of one sequence**. *Trap D (vocabulary size):* unrelated.

**Q3 → C. Query, Key and Value vectors.**
Each token's embedding is projected by three learned matrices into q, k and v; scores come from q·k and the output is the weighted sum of v: `softmax(QKᵀ/√d_k)V` (Vaswani p. 4).
*Trap A:* input/hidden/output are the generic names for any network layer, not the attention vectors. *Trap B and D:* invented names.

**Q16 → C. Their inherent sequential nature, which precludes parallelization within training examples.**
Near-quote from the paper: recurrent models process symbol by symbol, "precluding parallelization within training examples" (Vaswani p. 2).
*Trap A (cannot handle short-range dependencies):* reversed. RNNs handle short range fine; their weakness is long range and speed. *Trap B (needs external memory):* that is memory networks, a different model. *Trap D:* RNNs do not use grammar rules.

---

## B. Logit lens and layer dynamics

**Q4 → C. Analyse next-token prediction by applying the language-modelling head to intermediate hidden states (early decoding).**
The logit lens decodes the hidden state after each layer with the final output layer (`lm_head`) to see what the model "would predict now" (Lecture slide 21; Lab 9.3, p. 7).
*Trap A (reduce trainable parameters):* that is LoRA. *Trap B (GPU memory):* an engineering measurement. *Trap D (history list):* that is the chatbot task (Lab 9.2).

**Q5 → C. Argument Formation, Function Application and Saturation.**
The three stages from Merullo 2023: surface the argument, apply the function (an FFN update, about two-thirds of the way through), then the answer saturates (Lecture slide 22; Merullo pp. 3–4).
*Trap D (Tokenization, Embedding, Softmax):* these are steps of the pipeline around the model, not what happens to the prediction across layers. *Trap B (Pre-training, Fine-tuning, Deployment):* the model's life cycle, not its layers.

---

## C. Pre-training and NLP basics

**Q8 → B. Large-scale pre-training on general-domain data, and adaptation to particular tasks or domains.**
Wording from the LoRA abstract: "pre-training on general domain data and adaptation to particular tasks or domains" (Hu p. 1); the lecture shows the same two steps on slides 40–41.
*Trap C (human preference collection and reward-model training):* these are steps *inside* RLHF, which is one way to do the adaptation. *Trap A and D:* old rule-based, linguistics-driven NLP.

**Q11 → C. Self-supervised learning.**
Pre-training uses raw text where the **next token is the label**, so no human labels are needed (Lecture slides 11–12, 25).
*Trap A (supervised):* supervised learning needs human-provided labels; here the labels come from the text itself, which is why the data can be 10–35 trillion tokens (slide 10). *Trap B (reinforcement):* used later in RLHF, not for pre-training.

**Q12 → B. A representation of words as points in a high-dimensional vector space, allowing generalisation between related words.**
Russell & Norvig §25.1: represent words "as points in a high-dimensional space, rather than as atomic values", so related words get similar vectors (`Readings_W9_Russell&Norvig_C25.pdf`, pp. 1–2).
*Trap A (atomic, unique discrete values):* that is exactly **one-hot encoding**, which cannot capture similarity. *Trap C (tokenizing):* splitting text into pieces, a separate step. *Trap D (a Transformer layer that calculates attention):* wrong thing; embeddings are the input vectors.

**Q13 → D. All of the above.**
LLMs do well on question answering (Lecture slides 28–30 list Q&A and summarisation among the tasks), summarisation (Lab 9.1, your Gemini run) and are also strong at maths reasoning and code generation (not covered in the slides; general knowledge).
*Trap:* choosing a single true option when the question asks for the set.

---

## D. Prompting and chatbots

**Q14 → B. Prompt Engineering.**
Designing principles for writing prompts is prompt engineering (Lecture slides 32–36; Bsharat 2023: 26 principles).
*Trap A (PEFT):* changes model weights (LoRA), whereas prompting changes only the input. *Trap C and D:* RL and distillation are training methods.

**Q17 → C. To set the persona, tone or explicit behavioural constraints (e.g., "You are a helpful assistant").**
The system prompt is set by the developer and shapes overall behaviour; the user prompt asks for the specific output (Lecture slide 37; Lab 9.2, p. 5).
*Trap A (tokenization/positional encoding of the history):* done by the tokenizer and model internals. *Trap B (low-rank matrices):* LoRA. *Trap D (retrieve from pre-training memory):* no such mechanism.

**Q7 → A. The entire conversation history, concatenated with the current user query, must be passed back to the LLM for every new input.**
An LLM is stateless; its "memory" is the `messages` list re-sent every turn (Lab 9.2, p. 4–5 and the chatbot notebook).
*Trap B (re-train after each turn):* far too costly and unnecessary. *Trap C (a separate RNN compresses the history):* not what the lab does. *Trap D (delete the system prompt after turn 1):* the opposite, the system prompt stays at the start of the list so the behaviour persists.

**Q15 → B. Faster response speed but potentially less accurate or deep answers.**
Small model: fast and cheap but weaker knowledge; your Gemma-1B run said Brunel is in Bath (wrong) (Lab 9.2 requirement 4; chatbot notebook output).
*Trap A:* reversed (slower and better is the large model). *Trap D:* even a small model still needs the history.

---

## E. LoRA

**Q6 → C. Freeze the pre-trained weights and inject trainable rank-decomposition matrices, greatly reducing the number of trainable parameters.**
`h = W₀x + BAx`, with W₀ frozen and r ≪ d (Hu pp. 3–4; 10,000× fewer trainable parameters and 3× less GPU memory on GPT-3, p. 1).
*Trap B (full fine-tuning of all parameters):* what LoRA avoids. *Trap D (increase GPU memory):* LoRA **reduces** memory. *Trap A (replace attention with recurrence):* nothing in LoRA changes the architecture.

**Q18 → B. It freezes the pre-trained weights and injects small trainable rank-decomposition matrices (A and B) into the Transformer layers.**
Same mechanism as Q6 (Hu pp. 3–4).
*Trap A (update everything with a lower learning rate):* that is still full fine-tuning. *Trap C (pruning):* removes weights, a different technique. *Trap D (replace attention with a linear layer):* LoRA adds to the layers, it does not replace them.

---

## F. RLHF

**Q9 → C. The Reward Model.**
Trained on human-ranked data to predict good or bad responses (Lecture slide 52; Ziegler p. 3).
*Trap A (pre-trained language model) and B (self-supervised model):* those learn from raw text. *Trap D (instruction fine-tuning model):* learns from full example outputs, not rankings.

**Q19 → B. To predict the quality (good, bad or preference rank) of a generated response from human-labelled comparison data.**
Same as Q9. Ziegler's loss makes the human's chosen response get the highest score among four (Ziegler p. 3).
*Trap A (generate text):* that is the policy (the LLM itself). *Trap C (masked-language-model fine-tuning):* unrelated. *Trap D (Q·K scores):* attention.

**Q10 → B. Fine-tuning requires a full output for the target task, while RLHF primarily uses only a choice (e.g., ranking).**
Lecture slide 54: fine-tuning data = full output (hard), RLHF data = only a choice (easy).
*Trap A (fine-tuning unlabelled, RLHF fully labelled):* reversed. *Trap C:* both use human judgement; RLHF does not rely on simulated data. *Trap D:* mixes up pre-training data with fine-tuning data.

**Q20 → C. Only a choice or ranking (a human preference comparison) rather than a complete, perfect output.**
Same point as Q10 (Lecture slide 54; Ziegler p. 3: comparisons are easier and more consistent for humans than absolute scores).
*Trap B (a small set of full desired outputs):* describes **supervised** fine-tuning. *Trap A (unlabelled text):* pre-training. *Trap D (synthetic data):* not what the lecture says.

---

## Topic summary (one-liners to memorise)
- Transformer = attention only, **O(1) sequential steps**, Q/K/V, softmax(QKᵀ/√d_k)V.
- Logit lens = read each layer with the output head; stages **argument formation → function application → saturation**.
- Pre-training = **self-supervised** next-token prediction; then **adapt** (instruction fine-tuning, LoRA, RLHF).
- Word embedding = **point in vector space** (one-hot = atomic).
- Chatbot = **send the whole history** every turn; system prompt = persona/rules; small model = **fast but shallower**.
- LoRA = **freeze W₀, train small A and B**.
- RLHF = **reward model** trained on **choices/rankings**; fine-tuning needs **full outputs**.
