# 图外运行时：只用 onnxruntime，不装 PyTorch

读 `motif.onnx.export_onnx` 导出的 fp16 bundle（`decode.onnx` + 多张 `prefill_<T>.onnx` +
所有图共享的 `weights.bin`）做增量生成。
整层的依赖只有 `numpy` / `onnxruntime` / `tokenizers` / `transformers`，**没有任何 torch**，
连 tokenizer 也是库 `PackedTokenizer` 的无 torch 移植（`tokenizer.py`），不 import `motif.*`。
这一层的 HTTP 封装（OpenAI 兼容的 FastAPI 服务）在 `motif/service/`，它复用这里的 `template.py`，
所以服务与命令行渲染出的 prompt 逐 token 相同。

```
pip install codon-motif[onnx]
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `paths.py` | `default_bundle()` = 环境变量 `MOTIF_ONNX_BUNDLE`（未设则 `None`）；`default_tokenizer()` = `MOTIF_TOKENIZER`，否则当前目录的 `tokenizer.zip`，否则仓库根的 |
| `no_torch.py` | `block_torch()`：删掉已加载的 torch 模块并把 `sys.modules['torch']` 置为 `None`；`torch_blocked()` 查询状态 |
| `tokenizer.py` | 库 `PackedTokenizer` 的无 torch 移植：读 `tokenizer.zip`（`tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` / `safe_rules.json` / `safe_escape.txt`）、`encode` / `decode` / `token_to_id` / `vocab_size` / `token_eos` / `token_pad` / `apply_chat_template`（含结构字符转义） |
| `template.py` | 对话模板与推理段：`TOKEN_NAMES`、`EFFORT_ALIAS`、`resolve_effort`、`resolve_token_id`、`render_prompt`、`stop_token_ids`、`ThoughtSplit`、`split_thought` |
| `sampler.py` | `NumpySampler`：重复惩罚 → 温度 → top-k → top-p → softmax → 多项分布，顺序与 `motif.model.sampler.Sampler` 一致 |
| `engine.py` | `OrtEngine`：从图输入声明反推初始 cache（符号轴 = 0）；`prefill` 每步取**能塞下的最大档**（64 / 256 / 512 / 1024），某档抛异常就把它 drop 掉继续用更小的档；`step` 一步一 token，`generate` 采样到 stop 或 `max_new_tokens` |

命令行两条（留在研究目录，只作为这一层的调用样例）：
`.research/runtime/py/chat.py` 一次性 `--prompt` 或交互模式（`/reset`、`/exit`），打印
`[effort] [reasoning] [answer] [stats]`；`.research/runtime/py/check_runtime.py` 六段对照。

## 跑

```
python .research/runtime/py/chat.py --prompt '从前有座山，'
python .research/runtime/py/check_runtime.py                     # 六段对照
python -m motif.service.server --bundle .research/onnx/out/fp16 --no-torch
```

`chat.py --help` 看全部开关（`--effort`、`--no-think`、`--temperature`、`--top-k`、`--top-p`、
`--repetition-penalty`、`--max-new-tokens`、`--seed`、`--greedy`、`--bundle`、`--tokenizer`、`--system`）。
`chat.py` 在 import 这一层之前先调用 `no_torch.block_torch()`，所以生成过程中不可能偷偷用到 torch。

## 「真的没有 torch」是怎么证明的

`sys.modules['torch'] = None` 而不是装一个抛异常的 import hook：前者让 `import torch` 报
`ImportError: import of torch halted; None in sys.modules`，同时
`importlib.util.find_spec('torch')` 返回 `None`——`transformers` 内部正是这样探测 torch 的
（`transformers/utils/import_utils.py`），所以 `is_torch_available()` 会如实报 False。
换成一个会抛异常的 hook，`transformers` 自己的探测就会炸在 import 阶段。

`check_runtime.py` 第 6 段就是这件事的可执行证据：block 之后 `find_spec` 为 `None`、`import torch` 抛错、
tokenizer 仍能 encode/decode、引擎仍能生成。

## 已知限制

- **prefill 图是静态 T**（`export_onnx` 默认导出 64 / 256 / 512 / 1024 四档，`prefill_lengths=` 可换）。
  原因是 delta rule 的 `chunk_size` 要求 T 是 64 的倍数，`torch.export` 表达不了「倍数」约束
  （`((64 + (-1)*(T % 64)) % 64) != 0`）。prompt 不满最小一档或长尾由 decode 图逐 token 走完。
- **每张图在内存里各占一份权重**：五张图共享磁盘上的 `weights.bin`，但 onnxruntime 会给每个 session
  各复制一份初始值（约 328 MB/图）。同时持有全部档位约 1.6 GB；浏览器/Node 的 wasm 版
  放不下（见 `.research/runtime/js/README.md`），所以那边改成惰性建 session、放不下就 drop。
- **fp16 下不要用 torch 的 `temperature=0.0` 贪心**：库的采样器会把 logits 除以 `1e-5`，fp16 溢出成 inf → NaN。
  本运行时的 numpy 路径不受影响（内部用 float64），`--greedy` 同时把重复惩罚置为 1.0。
- 没移植 JSON 语法约束（库的 `motif/model/grammar.py`，它已随重构删除），也没移植 `generation_prompt_ids` / `strip_dialog_tokens`。
- 输出的质量取决于权重：`pretrained_base` 是 base 模型，没做后训练，所以正文通常不成句，推理段也常常空。

## 换成自己的权重

bundle 由 `motif.onnx.export_onnx` 生成：

```
python -m motif.onnx.export_onnx --weights D:\MotifChord\pretrained_base.safetensors `
    --precision fp16 --out .research/onnx/out/fp16 --prefill-lengths 64,256,512,1024
```

它按精度重新加载模型、导出 decode（T=1）与每档 prefill，把逐字节相同的 `weights.bin` 合并成一份。
导出后本运行时不需要改动，`--bundle`（或 `MOTIF_ONNX_BUNDLE`）指过去即可。

## 实测

`.research/runtime/py/check_runtime.py` 六段全绿（真实 172 M fp16 bundle，CPU）：

- 模板 id 与库完全一致、stop `(9, 23)`、effort 三个别名与报错文案一致。
- numpy vs torch 采样概率最差 **`4.317e-07`**（含 top-k/top-p 组合）。
- 教师强制 argmax **17/17 (100%)**、最差 `|diff| 2.734e-02`（fp16 两种内核的加法顺序不同）。
- 逐档 prefill（`position` 等于 chunk 长度）：64 → **651.2 ms**；256 → 1216.6 ms；512 → 3482.9 ms；
  1024 → 10176.8 ms（含首次建 session 的权重拷贝）。
- 端到端贪心 16 token 与 torch **id 完全相同**。
- 第 6 段把 torch 拿掉后：`find_spec` 为 `None`、`import torch` 抛
  `import of torch halted; None in sys.modules`、tokenizer 仍能 encode/decode、引擎仍能生成
  16 token（14.3 tok/s）。
