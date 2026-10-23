# Laya inference walkthrough

Everything below comes from running the published base model `convaiinnovations/laya` on CPU using a worked example below, and printing the numbers at each stage. It is the checkpoint the notebook starts from, not its fine-tuned output; the forward pass is identical, only the weight values differ. In Laya the question text is a fixed instruction and the state is the evidence, so I set it up as:

### Worked example

```python
state    = "what is the capital of France"
question = {"type": "choice",
            "instructions": "Which option correctly answers the question?",
            "criteria": ["Paris", "Vienna", "London"]}
```

Result: Paris 0.924, Vienna 0.047, London 0.029.

LLM writes an answer one token at a time, running the model once per token. Laya never writes anything but runs the model once over a sequence that already contains all three answers, then reads one number per answer and compares them. Its "response" is that comparison. This is why it is a decision engine: it can only rank options you give it.

## Step 1: Text becomes 26 integers

```
 0 [CLS]   1 choice  2 question  3 :  4 Which  5 option  6 correctly  7 answers  8 the  9 question  10 ?  11 [SEP]
12 [MASK] 13 Paris  14 [MASK] 15 Vienna  16 [MASK] 17 London  18 [SEP]
19 what   20 is     21 the      22 capital  23 of  24 France  25 [SEP]
```

Positions 12, 14 and 16 are the three answer slots. Everything that follows is about what ends up in those three positions.

## Step 2: Each integer becomes a vector

The first weights are a table of 50,368 rows × 1,024 numbers, one row per vocabulary entry (51.6M parameters). Token 6181 ("France") simply looks up row 6181.

At this stage the vectors carry no context at all. The three `[MASK]` vectors are exactly identical, since they are the same row of the same table; I checked. Whatever later makes the Paris slot different from the London slot has to come from the slots looking at their neighbours.

## Step 3: 28 encoder layers mix information between positions

The encoder is ModernBERT-large: 28 layers of about 12.3M weights each. Every layer does two things to the 26 × 1,024 grid of numbers, and adds the result to what was already there rather than replacing it. Each position's vector therefore works like a running notepad that the layers keep writing into.

Attention moves information between positions (about 4.2M weights per layer). For the slot at position 12:

- Three weight matrices turn every vector into a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I hand over?").
- The slot's query is compared with every token's key. The scores go through softmax and become mixing weights that sum to 1.
- The slot receives a weighted blend of the other tokens' values.

This happens in 16 parallel heads, each with its own 64-number slice and so its own idea of what is relevant. Position information is folded into the query–key comparison (rotary embeddings). That is how slot 12 can tell that "Paris" is the word right after it, not "London".

The feed-forward block transforms each position on its own (about 8.1M weights per layer). It expands the 1,024 numbers to 2,624 features, switches some on and some off (a gated nonlinearity), and compresses back. Interpretability work on similar models (Meng et al., 2022, on GPT) finds that much factual association, such as "France → Paris", is stored in these blocks: a pattern in the input switches on features that write a related pattern into the output.

Here is how far apart the three slot vectors drift, measured as cosine similarity (1 = same direction, 0 = unrelated, negative = opposite):

| After layer | Paris vs Vienna | Paris vs London | Vienna vs London |
|---|---|---|---|
| 0 (embeddings) | 1.000 | 1.000 | 1.000 |
| 7 | 0.947 | 0.947 | 0.985 |
| 14 | 0.935 | 0.924 | 0.972 |
| 21 | 0.755 | 0.656 | 0.967 |
| 28 | −0.532 | −0.656 | 0.909 |

The decision shows up as geometry. The slots start identical, take on their own option's flavour, and in the last third of the network the Paris slot swings to point roughly the opposite way from the other two. Vienna and London stay close together as the two wrong answers.

Attention maps are less clean. Averaged over the 16 heads, the slots attend mostly to "the" and `[SEP]` and give "France" about 1% of their attention. Those filler tokens are known to act as gathering points, where earlier layers have already pooled the sentence's content. So the "France → Paris" signal reaches the slot indirectly and through particular heads, and averaging hides it. You won't find one arrow from "France" to "Paris" in this model.

## Step 4: Laya's own layers

Everything so far is the pre-trained language model. Laya then adds:

1. A question-type vector, one of 3 learned rows, added to every position. It tells the rest of the network that this is a `choice` question, not a `score` or yes/no one.
2. Two more transformer layers of the same kind (25.2M weights), trained for Laya rather than inherited. They let the three slots, which by now each hold a summary of "my option in this context", compare against each other before scoring.

## Step 5: The scorer turns each slot into one number

The scorer is a small network (about 1.05M weights): normalise → multiply by a 1,024 × 1,024 matrix and add a bias → nonlinearity → multiply by a 1,024-number vector and add a bias. It applies exactly the same weights to each of the three slots. It has no idea which option is which; it is one learned detector for "this vector looks like a correct answer".

| Option | Logit |
|---|---|
| Paris | 4.070 |
| Vienna | −1.177 |
| London | −2.026 |

About 240 of the 1,024 hidden features fire for each slot. The 6.1-point gap between Paris and London is carried mainly by about 25 features, and the top 10 account for 4.9 of it. A broad, distributed representation gets read out through a small number of directions at the very end.

On "weights and biases": the ModernBERT encoder, 94% of the model, has no biases at all; its configuration turns them off. Biases appear only in Laya's added layers and the scorer.

## Step 6: Numbers become probabilities

Softmax raises e to each logit and divides by the total, so the results are positive and sum to 1.

- Raw: Paris 0.9925, Vienna 0.0052, London 0.0022.
- Calibrated: the checkpoint divides the logits by a fitted temperature of 1.76 (for choice questions with 3–5 options), giving 2.31, −0.67 and −1.15. The final answer is Paris 0.924, Vienna 0.047, London 0.029.

This is the calibration step from the notebook. The raw model was overconfident, and dividing by the temperature tones it down without changing the ranking.

## Where the knowledge lives: probing it

I changed one input at a time and re-ran the model:

| State | Paris | Vienna | London |
|---|---|---|---|
| what is the capital of France | 0.924 | 0.047 | 0.029 |
| what is the capital of Austria | 0.009 | 0.980 | 0.011 |
| what is the capital of the United Kingdom | 0.050 | 0.158 | 0.792 |
| what is the capital of Germany | 0.119 | 0.797 | 0.084 |
| what is the largest city in France | 0.965 | 0.022 | 0.012 |
| (empty) | 0.489 | 0.297 | 0.214 |
| capital of France, options reordered to London, Vienna, Paris | 0.856 | 0.102 | 0.043 |

- The knowledge really is in the weights. Change one word and the winner flips, using nothing but the same frozen weights.
- It works by association, not by looking up a fact. Asked about Germany, with Berlin not on offer, the model confidently picks Vienna, the most German-like option. "Largest city in France" gives Paris more confidently than "capital" does. The model strongly associates France with Paris, not "capital-of" with a particular city.
- Softmax always sums to 1, so the model must spread 100% over the options it is given and cannot say "none of these". (Laya's separate escalate head exists for that.)
- With no evidence at all it still leans towards Paris (0.49), and moving Paris to the last slot costs it 7 points. Those are biases built into the weights: a preference for the first position and familiar words, the position effect the library's option-shuffling feature is meant to counter.

## Where it was learned, and what training changes

The facts came from ModernBERT's pre-training, around 2 trillion tokens of text. Its task was to fill in hidden words: shown "The capital of France is [MASK]", it was graded on whether it predicted "Paris". After billions of such corrections, its weights make a `[MASK]` near "capital … France" carry Paris-like content. Laya reuses exactly that trick: each answer slot is a `[MASK]`, so the model asks for each slot, "how well does this option fit the gap, given everything around it?"

The training in the fine-tuning notebook doesn't install new facts on that scale. Each step computes the gap between predicted and gold probabilities and, via the gradient, works out for every one of the 421M numbers whether a tiny increase or decrease would shrink that gap, then nudges each one slightly. Over about 5,600 items × 4 epochs, this mostly retunes the final geometry: which directions the scorer reads, and how the last layers arrange the slots. The model's judgements then match the reference labels on this kind of case.