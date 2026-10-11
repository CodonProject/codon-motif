# Motif Chord Language Model

[English](#motif-chord-language-model) | [中文](https://github.com/CodonProject/codon-motif/blob/main/README.zh-CN.md)

Motif Chord is a small, CPU-friendly decoder-only causal language model with a mixed attention trunk: low memory use and high compute efficiency at long context.

- **Gated Delta layers** carry the bulk of the sequence with a linear-time chunked recurrence, dropping the memory cost of long sequences dramatically.
- **Chord layers** bring exact softmax attention and fuse a gated geometric-decay bias into the attention scores; a handful of parameters buys a Bayesian positional prior and replaces RoPE.
- **Attention residual** abandons the fixed additive residual inside a Block and instead aggregates the outputs of earlier layers with learnable pseudo-queries. This eases the dilution of information in deep stacks, letting every layer selectively pull the features it needs from the layers below.

### Architecture preview

Gated Delta and Chord layers form one Block at a default 3:1 ratio, which balances long-text efficiency against exact retrieval. The overall structure:

![model](https://raw.githubusercontent.com/CodonProject/codon-motif/main/asset/model.png)

#### 1. Hybrid attention

To balance efficiency against precision the model uses a 3:1 mixed-attention ratio:

- **Gated DeltaNet** performs 75% of the work in a Block. Linear-time chunked recurrence plus a short causal convolution (Conv1d + SiLU) cuts the memory footprint and inference cost of long sequences, which suits edge and CPU environments.
- **Chord Attention** inserts exact softmax attention every 4 layers for global, precise retrieval. The ratio follows the engineering recipe of the Qwen 3 family.

#### 2. Bayesian distance prior

For positional encoding the project drops RoPE and follows the Bayesian Attention Mechanism instead.

- A gated geometric-decay bias is added to the attention scores.
- `alpha`, `beta` and `mu` are learnable per attention head (`mu` is frozen), so position information becomes a Bayesian prior inside the attention scores rather than an add-on encoding.
- This design explores length extrapolation with very few extra parameters.

#### 3. Intra-block attention residual

To ease information dilution in deep stacks the model introduces Attention Residuals along the depth dimension (following the Kimi team's paper).

- **Engineering trade-off**: limited by single-GPU memory and a 172M model, this project implements the lightweight **Intra-Block** version rather than global aggregation across Blocks.
- **Mechanism**: inside each `DecoderBlock`, every layer aggregates the outputs of the previous layers through learnable `pseudo_queries` combined with `RMSNorm`, routing information across layers on demand.

#### 4. Overall data flow

The model stacks several `DecoderBlock`s. Each Block holds 3 Gated Delta Decoders and 1 Chord Decoder, uses a parallel Attention-MLP residual (`x = x + attn(x) + mlp(x)`), and passes state between layers through AttnRes.

### Quick start

#### Install

```bash
pip install codon-motif
```

For an editable install from source (also pulls in the test dependencies):

```bash
pip install -e ".[dev]"
```

#### Load a checkpoint

```python
from motif.model import MotifChordModel

model = MotifChordModel.from_pretrained('MotifChord-Base.safetensors')
```

`from_pretrained` accepts a safetensors file or shard directory, a bare `state_dict` checkpoint, and the pipeline checkpoints this project's trainer writes.

#### Generate

```python
from motif.model import Sampler, TextGeneration
from motif.tokenizer.tokenizer import PackedTokenizer

tokenizer = PackedTokenizer('tokenizer.zip')
model = MotifChordModel.from_pretrained('MotifChord-Base.safetensors').eval()

generation = TextGeneration(
    model,
    tokenizer,
    'The capital of France is',
    max_new_tokens=48,
    sampler=Sampler(temperature=0.8, top_k=50),
)
print(generation.run().text)
```

#### Chat

Chat needs a post-trained checkpoint, and post-training is not finished yet.

### Training details

Limited to a single GPU, all experiments ran on one 32 GB card (5090).

- **Parameters**: 172.3 M
- **Precision**: Bfloat16 / Float32

#### 1. Pretraining

The model was pretrained on 13.78B tokens over 4,042,962 steps, using a progressive long-context schedule:

| Stage         | Sequence Length | Batch Size | Tokens | Steps     |
| :------------ | :-------------- | :--------- | :----- | :-------- |
| Foundation    | 1024            | 4          | 5.77B  | 1,877,825 |
| Expansion     | 2048            | 2          | 0.86B  | 422,048   |
| Stabilization | 4096            | 1          | 7.14B  | 1,743,089 |

Training stayed stable throughout: the final cross-entropy loss converged around 2.8 with no visible loss spikes, which is the numerical-stability evidence for AttnRes plus the 3:1 hybrid attention recipe on a single card.

![loss_fg](https://raw.githubusercontent.com/CodonProject/codon-motif/main/asset/loss_fg.png)

### Acknowledgements

This project is an exploration of mixed-attention architectures on a single GPU, and it would not exist without the open-source community's research and high-quality datasets.

**Particular thanks to the open datasets that provided the pretraining corpus:**

- **UltraData-Code** (OpenBMB, 2026): large-scale code data that helped the model build basic logical structure.
- **arXiv Complete** (secemp9, 2026): arXiv metadata and document snapshots that injected academic text.
- **WMT17 Zh-En** (Huang et al., 2025): Chinese-English parallel data that markedly improved bilingual alignment.
- **Nemotron-CC-Math-v1** (NVIDIA, 2025): high-quality mathematical reasoning text that reinforced the science side.
- **Ultra-FineWeb-L3** (OpenBMB, 2026): high-quality web text that underpins the model's generalisation.

**Architecture and theory:**

- **Gated DeltaNet** (Yang et al., 2025): the core linear attention mechanism, making O(N) inference and recurrent state caching possible.
- **Kimi Attention Residuals** (Kimi Team, 2026): inspired the intra-block dynamic aggregation in `DecoderBlock`.
- **Bayesian Attention Mechanism (BAM)** (Bianchessi et al., 2026): introduced the generalised Gaussian prior that replaced RoPE in the Chord Decoder.
- **Qwen3 Technical Report** (Qwen Team, 2025): provided the 3:1 hybrid-attention engineering recipe.

**Development model and tools:**

Thanks to the open-source community (PyTorch, Triton, HuggingFace and more) for the tooling. This project used a **"human-led architecture design + AI-assisted low-level implementation"** model: the cache management, the AttnRes tensor-lifetime planning and the architecture alignment were done by the author, while much of the heavy kernel work (such as the fused Triton operators) was generated with AI coding tools and then debugged and performance-tuned rigorously by the author.

Finally, thanks to every developer who keeps exploring architectural boundaries under limited compute. As a 172M single-GPU project it may never top a leaderboard, but it shows that **with an interesting enough architectural idea and a little engineering patience, you can still verify the frontier of academic papers yourself on limited hardware.**

<details>
<summary>Citations and references</summary>

```bibtex
% 1. Gated DeltaNet
@inproceedings{yang2025gated,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Yang, Songlin and Kautz, Jan and Hatamizadeh, Ali},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  eprint    = {2412.06464},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2412.06464}
}

% 2. Kimi Attention Residuals
@misc{kimiteam2026attentionresiduals,
  title         = {Attention Residuals},
  author        = {Kimi Team and others},
  year          = {2026},
  eprint        = {2603.15031},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.15031}
}

% 3. Bayesian Attention Mechanism
@inproceedings{bianchessi2026iclr-bayesian,
  title     = {Bayesian Attention Mechanism: A Probabilistic Framework for 
               Positional Encoding and Context Length Extrapolation},
  author    = {Bianchessi, Arthur S. and Aguirre, Yasmin C. and 
               Barros, Rodrigo C. and Kupssinsk{\"u}, Lucas S.},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  eprint    = {2505.22842},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2505.22842}
}

% 4. Qwen3-Next / Qwen3 Technical Report
@misc{qwen3technicalreport,
  title         = {Qwen3 Technical Report},
  author        = {Qwen Team},
  year          = {2025},
  eprint        = {2505.09388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2505.09388}
}

% 5. UltraData-Code
@misc{openbmb_ultradata_code,
  title={UltraData-Code},
  author={OpenBMB},
  year={2026},
  url={https://huggingface.co/datasets/openbmb/UltraData-Code}
}

% 6. arXiv Complete
@misc{secemp9_arxiv_complete,
  title={arXiv Complete: A Snapshot of arXiv's Metadata, Version History, Submission Files and Rendered Documents},
  author={secemp9},
  year={2026},
  url={https://huggingface.co/datasets/secemp9/arxiv-complete}
}

% 7. WMT17 Zh-En
@dataset{fei_huang_2025_5g3elaik,
  author       = {Fei Huang and Hao Zhou and Yang Liu and Hang Li and Minlie Huang},
  title        = {WMT17 Zh-En},
  month        = jan,
  year         = 2025,
  publisher    = {No Organization},
  doi          = {10.57702/5g3elaik},
  url          = {https://doi.org/10.57702/5g3elaik}
}

% 8. Nemotron-CC-Math-v1
@misc{nvidia_nemotron_cc_math_v1,
  title={Nemotron-CC-Math-v1},
  author={{NVIDIA}},
  year={2025},
  url={https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1}
}

% 9. Ultra-FineWeb-L3
@misc{ultra-fineweb-l3,
  title={Ultra-FineWeb-L3},
  author={OpenBMB},
  year={2026},
  url={https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3},
  publisher={Hugging Face}
}
```

</details>

---

## Repository and development

Model weights ship as release assets, not as commits. The test suite is self-contained: it trains its own tiny tokenizer and builds small models, so no checkpoint download is required.

```bash
pytest
```

`examples/` holds runnable scripts (`verify.py` checks the environment and weights first, `chat.py` talks to the model, `generate.py` continues text, `pretrain.py` and `sft.py` train). They need `tokenizer.zip` in the repository root or in `../`. Temporary artefacts (corpora, checkpoints, the Triton kernel cache) are written under `<repo>/.tmp` and ignored.

Apache-2.0. See `LICENSE`. Intended use, training, evaluation and limitations are in [`MODEL_CARD.md`](https://github.com/CodonProject/codon-motif/blob/main/MODEL_CARD.md).
