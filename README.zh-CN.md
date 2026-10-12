# Motif Chord 语言模型

[English](https://github.com/CodonProject/codon-motif/blob/main/README.md) | [中文](#motif-chord-语言模型)

Motif Chord 是一个对 CPU 友好的小型 decoder-only 因果语言模型，采用混合注意力机制，在长上下文中兼顾了内存占用低与计算效率高：

- **Gated Delta 层**：以线性时间的分块递推承担序列的主要部分，大幅降低长序列的显存消耗。
- **Chord 层**：引入精确的 softmax 注意力，并在注意力分数上融合了带门控的几何衰减偏置，仅引入极少参数便实现了贝叶斯位置先验，替代了传统的 RoPE。
- **注意力残差**：在 Block 内部摒弃了传统的固定权重加性残差，转而利用可学习的伪查询动态聚合历史层的输出。这种机制有效缓解了深层网络中的信息稀释问题，让每一层都能选择性地提取前序层的有用特征。

### 架构预览

Gated Delta 层与 Chord 层默认使用 3:1 的比例组成一个 Block，以在长文本效率与精确检索能力之间取得最优平衡。整体结构如下图所示：

![model](https://raw.githubusercontent.com/CodonProject/codon-motif/main/asset/model.png)

#### 1. 混合注意力

为了在效率与精度之间取得平衡，模型采用了 3:1 的混合注意力配比：

- **Gated DeltaNet**：承担 Block 中 75% 的计算量。通过线性时间的分块递推和短因果卷积（Conv1d + SiLU），极大地降低了长序列的显存占用和推理成本，非常适合端侧或 CPU 环境。
- **Chord Attention**：每 4 层插入一次精确的 Softmax 注意力，提供全局的精确检索能力。该配比参考了 Qwen 3 系列的架构经验。

#### 2. 贝叶斯距离先验

在位置编码方面，本项目摒弃了传统的 RoPE，转而采用 Bayesian Attention Mechanism 的思路。

- 在注意力分数上引入了带门控的几何衰减偏置。
- `alpha`、`beta`、`mu` 被设计为每个注意力头可学习的参数（其中 `mu` 被冻结），使位置信息内化为注意力分数的贝叶斯先验，而非外挂式编码。
- 这种设计仅引入极少参数，便探索了模型在长度外推上的泛化潜力。

#### 3. 块内注意力残差

为了缓解深层网络中的信息稀释问题，模型在深度维度引入了 Attention Residuals（参考 Kimi 团队的论文）。

- **工程权衡**：受限于单卡显存和 172M 的模型规模，本项目实现的是 **Intra-Block（块内）** 的轻量级版本，而非跨 Block 的全局聚合。
- **机制**：在每个 `DecoderBlock` 内部，每一层通过可学习的 `pseudo_queries` 结合 `RMSNorm`，动态聚合前序层的输出。这有效实现了跨层信息的按需路由。

#### 4. 整体数据流

模型由多个 `DecoderBlock` 堆叠而成。每个 Block 内部包含 3 个 Gated Delta Decoder 和 1 个 Chord Decoder，采用并行的 Attention-MLP 残差结构（`x = x + attn(x) + mlp(x)`），层与层之间通过 AttnRes 机制进行历史状态的信息流传递。

### 快速开始

#### 安装

```bash
pip install codon-motif
```

从源码可编辑安装（同时会装上测试依赖）：

```bash
pip install -e ".[dev]"
```

#### 加载 checkpoint

```python
from motif.model import MotifChordModel

model = MotifChordModel.from_pretrained('MotifChord-Base.safetensors')
```

`from_pretrained` 接受 safetensors 单文件或分片目录、裸 `state_dict` checkpoint，以及本项目训练器写出的 pipeline checkpoint。

#### 生成

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

#### 对话

对话依赖后训练模型检查点，当前还没有完成模型的后训练。

### 训练细节

本项目受限于单卡算力，所有实验均在单卡 32G 显存（5090）上完成。

- **模型参数**: 172.3 M
- **训练精度**: Bfloat16 / Float32

#### 1. 预训练

模型在 13.78B tokens 上完成了预训练，总计 4,042,962 steps。采用了渐进式的长上下文训练策略：

| Stage         | Sequence Length | Batch Size | Tokens | Steps     |
| :------------ | :-------------- | :--------- | :----- | :-------- |
| Foundation    | 1024            | 4          | 5.77B  | 1,877,825 |
| Expansion     | 2048            | 2          | 0.86B  | 422,048   |
| Stabilization | 4096            | 1          | 7.14B  | 1,743,089 |

训练全程平稳，最终 Cross-Entropy Loss 收敛于 2.8 附近，无明显的 Loss Spike 现象，证明了 AttnRes + 3:1 混合注意力架构在单卡训练下的数值稳定性。

![loss_fg](https://raw.githubusercontent.com/CodonProject/codon-motif/main/asset/loss_fg.png)

### 致谢

本项目是一次在单卡上进行的混合注意力架构探索，离不开开源社区优秀研究成果与高质量数据集的支撑。

**特别感谢以下开源数据集为模型预训练提供的宝贵语料：**

- **UltraData-Code** (OpenBMB, 2026)：提供了大规模的代码数据，帮助模型建立基础的逻辑结构能力。
- **arXiv Complete** (secemp9, 2026)：提供了 arXiv 论文的元数据与文档快照，为模型注入了学术语料。
- **WMT17 Zh-En** (Huang et al., 2025)：提供的中英平行语料显著提升了模型的双语对齐能力。
- **Nemotron-CC-Math-v1** (NVIDIA, 2025)：提供了高质量的数学推理语料，强化了模型在理科方向的文本理解。
- **Ultra-FineWeb-L3** (OpenBMB, 2026)：提供的高质量网页语料构成了模型泛化能力的基础。

**架构与理论支撑：**

- **Gated DeltaNet** (Yang et al., 2025)：为模型提供了核心的线性注意力机制，使得 O(N) 复杂度的推理与递归状态缓存成为可能。
- **Kimi Attention Residuals** (Kimi Team, 2026)：启发了 `DecoderBlock` 中的块内动态聚合机制。
- **Bayesian Attention Mechanism (BAM)** (Bianchessi et al., 2026)：引入了广义高斯分布先验，替代了 Chord Decoder 中的 RoPE。
- **Qwen3 Technical Report** (Qwen Team, 2025)：提供了 3:1 混合注意力比例的工程配方。

**开发模式与工具：**

感谢开源社区（PyTorch、Triton、HuggingFace 等）提供的强大工具。本项目采用了 **“人类主导架构设计 + AI 辅助底层实现”** 的开发模式：系统的缓存管理、AttnRes 张量生命周期规划与架构对齐由作者亲自完成，而底层繁重的算子实现（如 Triton 融合算子）大量使用了 AI 编程工具辅助生成，随后由作者进行了严格的调试与性能权衡。

最后，感谢所有在有限算力下依然坚持探索架构边界的开发者们。作为一个 172M 参数的单卡项目，它可能不会在榜单上霸榜，但它证明了：**只要有足够有趣的架构想法和一点工程耐心，你依然可以在有限的算力下，亲手验证最前沿的学术论文。**

<details>
<summary>引用与参考文献</summary>

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

## 仓库与开发

模型权重以 release asset 发布，不提交进仓库。测试套件自给自足：它自己训练一个极小的 tokenizer 并搭建小模型，不需要下载任何 checkpoint。

```bash
pytest
```

`examples/` 下有可直接运行的脚本（`verify.py` 先自检环境与权重，`chat.py` 对话，`generate.py` 续写，`pretrain.py` / `sft.py` 训练）；运行需要仓库根或 `../` 下的 `tokenizer.zip`。临时产物（语料、checkpoint、Triton kernel 缓存）写在 `<repo>/.tmp` 下并被忽略。

模型本体之外还有两个可选包，各自带依赖，所以不进核心安装：

```bash
pip install "codon-motif[onnx]"      # 导出 ONNX，并在不装 PyTorch 的情况下推理
python -m motif.onnx.export_onnx --weights MotifChord-Base.safetensors --out out/fp16

pip install "codon-motif[service]"   # 用 OpenAI 兼容格式部署这份 bundle
python -m motif.service.server --bundle out/fp16 --no-torch
```

`motif.onnx` 导出一张 decode 图、每个档位一张 prefill 图，五张图共享同一份 `weights.bin`；`motif.onnx.runtime` 只用 `onnxruntime` 执行它们——不需要 `torch`，自带 tokenizer。`motif.service` 在此之上提供 `/v1/models`、`/v1/chat/completions`（流式与非流式）、`/v1/completions` 和 `/health`；默认走 ONNX 运行时，只有在传入 model 实例或 checkpoint 时才导入 `motif`。

Apache-2.0，见 `LICENSE`。用途、训练、评测与局限性见 [`MODEL_CARD.md`](https://github.com/CodonProject/codon-motif/blob/main/MODEL_CARD.md)（英文）。
