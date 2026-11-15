# TPT

A small GPT-style language model trained from scratch, and a mixture-of-experts version of it, compared on the same data.

## What is in here

- `model.py`: the transformer and the MoE layer
- `train.py`: training loop with checkpoint and resume
- `data.py`: reads the tokenized `.bin` files
- `evaluate.py`: validation loss, perplexity, bits per token and bits per byte
- `generate.py`: sampling

## Data and tokenizer

Text comes from FineWeb-Edu (65%) and Wikipedia (35%). The tokenizer is byte-level BPE with a vocabulary of 16,384, trained on the training split only. It averages [x] bytes per token on the validation set.


## Models

| | Dense | MoE |
|---|---|---|
| Layers / heads / width | 8 / 8 / 1024 | 8 / 8 / 768|
| Experts | none | 8, top-2 |


## Training

Both runs used the same data order, token budget, context length (512), learning rate schedule, seed and GPU.

| | Dense | MoE |
|---|---|---|
| Tokens seen | 2B | 2B |
| Peak learning rate | 3e-4 | 3e-4 |
| GPU hours (A100) | 3hrs | 5hrs |
| Peak memory | 65 GB | 73 GB |

## Results

All numbers are on the held-out validation set.

| | Loss (nats) | Perplexity |
|---|---|---|
| Dense | 3.4 |30| 
| MoE | 3.2 | 24 |

Perplexity is exp(loss).

[Two or three sentences on what you found, including anything that did not work.]

![loss curves](loss_curves.png)

## Expert load

Share of tokens routed to each expert, [with / without] the load-balancing loss.

![expert load](expert_load.png)

[One sentence: did routing collapse, and what did the balancing loss change?]

## Samples

Prompt: "Paris is the capital and most populous city of"

> Dense: Paris is the capital and most populous city of the state.
        The city is located at , with a population density of . The town has an average elevation of . The population density includes the following towns:
        - Kinx Falls, Shanghai
        - Wujiang Yi (Singinglong)
        - Tongyang (Tongyang)
        The capital is Guangzhou and Shengao Jhua. There are four major cities in China and a total of  is the city's capital, Chennai (Linglong) and Kunbu (Kand 

> MOE: Paris is the capital and most populous city of the U.S., but it’s also a major      commercial center in southern Washington, where it is the largest and biggest shopping district in the country.
The Miami Heights neighborhood (named after Hilary Clinton) has been one of the world’s richest cities for over two decades since its founding. Today, it’s a popular spot to visit and watch television shows as well as movie clips by many people throughout the United States.
There are more than 750 square miles in Washington in the U.S., according to the Miami Heights International 


Temperature 0.8, top-k 40.

## Limits

The models are small and trained on 2B tokens, so the text is often wrong or repetitive. 



## References

GPT-2 (Radford et al.), Switch Transformer (Fedus et al.), Mixtral (Jiang et al.)