import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from motif.model.generate import TextGeneration
from motif.pipeline.base import move_to_device, resolve_eos_token_id, select_model_kwargs
from motif.pipeline.callback import Callback
from motif.pipeline.probe import build_generate_kwargs


#: Label value `MotifSFT` uses for "this position is not supervised".
IGNORE_INDEX = -100


@dataclass
class ProbeSample:
    '''
    One fixed sanity sample, already rendered the way training renders it.

    Attributes:
        name (str): Short label used in the report (`#12`, or `stage0#12` when the run
            trains on several stages).
        input_ids (torch.Tensor): `[T]` full conversation, padded to the dataset's
            `pad_length`.
        labels (torch.Tensor): `[T]` supervision mask, `-100` outside the model turns.
        attention_mask (torch.Tensor): `[T]` `1` on real tokens, `0` on the padding.
        prompt_ids (torch.Tensor): `[T']` conversation up to the held-out model turn,
            ending on `<|im_start|><|model|>` when there is one.
        answer (str): The held-out model turn, when the sample had one.
    '''

    name: str
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    prompt_ids: torch.Tensor
    answer: str = ''


def _pick_indices(total: int, count: int) -> List[int]:
    '''
    Spread `count` indices evenly over `range(total)`, deterministically.

    The same groups are picked at every probe, so the loss curve is comparable step to
    step, and the spread walks the whole corpus instead of only its head.

    Args:
        total (int): Number of candidates; must be positive.
        count (int): Number of indices to pick, clamped to `total`.

    Returns:
        List[int]: Sorted, duplicate-free indices.
    '''
    count = max(1, min(int(count), total))
    return sorted({min(total - 1, int(round(k * total / count))) for k in range(count)})


def build_probe_samples(datasets, count: int = 2) -> List[ProbeSample]:
    '''
    Pick `count` fixed samples out of one or more `MotifSFT` datasets.

    Every candidate is a conversation group of some dataset; the groups of all the
    datasets (one per SFT stage) are pooled and then spread evenly, so nothing is
    silently left unprobed. A sample whose last model turn answers a real turn holds that
    turn out as the expected answer; anything else (a single model turn, a conversation
    without a reply yet) is probed as a plain continuation.

    Args:
        datasets: A `MotifSFT`, or a sequence of them (one per stage).
        count (int): Number of samples to build, clamped to the number of groups.

    Returns:
        List[ProbeSample]: The rendered samples.

    Raises:
        ValueError: If `count` is not positive, if a dataset does not expose the render
            seam (`groups` / `build_messages` / `render_group` / `session_for`), or if
            nothing usable could be built.
    '''
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    datasets = [dataset for dataset in datasets if dataset is not None]
    if int(count) <= 0:
        raise ValueError(f'count must be positive, got {count!r}')

    entries: List[Tuple[int, Any, int, dict]] = []
    for dataset_index, dataset in enumerate(datasets):
        for attribute in ('groups', 'build_messages', 'render_group', 'session_for'):
            if not hasattr(dataset, attribute):
                raise ValueError(
                    f'the sanity probe needs a MotifSFT-like dataset exposing '
                    f'`{attribute}`, got {type(dataset).__name__}'
                )
        for group_index, group in enumerate(getattr(dataset, 'groups') or []):
            entries.append((dataset_index, dataset, group_index, group))
    if not entries:
        raise ValueError('the sanity probe found no conversation group to probe')

    samples: List[ProbeSample] = []
    for flat_index in _pick_indices(len(entries), count):
        dataset_index, dataset, group_index, group = entries[flat_index]
        messages = dataset.build_messages(group)
        if not messages:
            continue

        answer = ''
        holds_a_reply = (
            len(messages) >= 2
            and messages[-1].get('role') == 'model'
            and messages[-2].get('role') != 'system'
        )
        if holds_a_reply:
            answer = str(messages[-1].get('content') or '')
            prompt_messages = messages[:-1]
        else:
            prompt_messages = messages

        tensors = dataset.render_group(group)
        prompt_session = dataset.session_for(prompt_messages)
        if answer:
            prompt_session.add_generation_prompt()
        prompt_ids = prompt_session.to_tensors()['input_ids']
        pad_length = int(getattr(dataset, 'pad_length', 0) or 0)
        if pad_length and prompt_ids.numel() > pad_length:
            prompt_ids = prompt_ids[-pad_length:].contiguous()
        if prompt_ids.numel() == 0:
            continue

        name = f'#{group_index}' if len(datasets) == 1 else f'stage{dataset_index}#{group_index}'
        samples.append(ProbeSample(
            name=name,
            input_ids=tensors['input_ids'],
            labels=tensors['labels'],
            attention_mask=tensors['attention_mask'],
            prompt_ids=prompt_ids,
            answer=answer,
        ))

    if not samples:
        raise ValueError('the sanity probe found no usable conversation to probe')
    return samples


def _short(text: str, limit: int = 160) -> str:
    '''
    One-line version of a generated reply, for the log.

    Args:
        text (str): Generated text.
        limit (int): Maximum number of characters kept.

    Returns:
        str: The text with newlines escaped and truncated.
    '''
    flat = text.replace('\n', '\\n')
    return flat if len(flat) <= limit else flat[:limit] + '...'


def _write(text: str) -> None:
    '''
    Print one line of probe output without ever raising.

    A generated reply is arbitrary text: on a console that cannot encode it (a GBK
    terminal meeting a `\\ufffd` from a byte-level tokenizer, for instance) `tqdm.write`
    would raise `UnicodeEncodeError` - and a report that crashes is a probe that brings
    training down, which is exactly what a sanity check must not do. Characters the
    output encoding cannot represent are replaced before printing.

    Args:
        text (str): The line to print.
    '''
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, 'replace').decode(encoding, 'replace')
    try:
        tqdm.write(text)
    except Exception:
        pass


def _perplexity(loss: float) -> float:
    '''
    `exp(loss)`, guarded against overflow on a diverged loss.

    Args:
        loss (float): Cross-entropy loss.

    Returns:
        float: The perplexity, or infinity when the loss is not finite / too large.
    '''
    if not math.isfinite(loss) or loss >= 50.0:
        return float('inf')
    return math.exp(loss)


def _repeats(ids: Sequence[int], span: int = 4, threshold: int = 4) -> bool:
    '''
    Whether some `span`-token window occurs at least `threshold` times in `ids`.

    A cheap loop detector: a model that collapsed onto one phrase repeats a short window
    far more often than a piece of ordinary text does.

    Args:
        ids (Sequence[int]): Generated token ids.
        span (int): Window length in tokens.
        threshold (int): Number of occurrences that counts as a loop.

    Returns:
        bool: True when a window repeats `threshold` times or more.
    '''
    if len(ids) < span * threshold:
        return False
    counts: Dict[Tuple[int, ...], int] = {}
    for index in range(len(ids) - span + 1):
        window = tuple(int(token) for token in ids[index:index + span])
        counts[window] = counts.get(window, 0) + 1
        if counts[window] >= threshold:
            return True
    return False


class SanityProbe(Callback):
    '''
    Score fixed corpus samples and generate from them every `every_steps` steps.

    The samples are built once, outside the training loop (`build_probe_samples`), so
    the reported numbers are comparable across steps: the probe never touches the
    dataset stream the optimiser is consuming.

    This is the chat-formatted counterpart of `motif.pipeline.probe.Probe`, which
    generates from a raw text prompt (`tokenizer.encode('The')`): that is right for a
    pre-trained model but not for SFT, because the fine-tuned model has only ever seen
    *chat-formatted* conversations, so a bare string reaches it in a shape it does not
    know. Every sample here is a real conversation of the corpus, rendered through
    exactly the path training uses (the dataset's own `build_messages()` /
    `render_group()`), so the probe measures the very tokens the optimiser measures:

    * **teacher-forced score** - loss / perplexity and next-token accuracy over the
      supervised positions only (`labels != ignore_index`, that is the model turns). It
      is a number comparable from one probe to the next, unlike a sampled reply;
    * **generation** - the conversation up to its last model turn is rendered through
      the chat template and completed, so the report shows the model writing in the
      shape it was trained for;
    * **verdicts** - a handful of cheap checks (a loss that is not finite, a sample with
      no supervised token at all, an empty / immediately-ended / looping reply, a mean
      loss that grew by more than `warn_ratio` since the first probe) are reported
      loudly.

    Like `Probe`, the sanity probe only warns - it must never bring training down - and
    it runs on rank 0 only (`Callback.main_process_only`). Its scalars (`sanity/loss`,
    `sanity/ppl`, `sanity/acc`, `sanity/supervised`, `sanity/warnings`) are merged into
    the step metrics, so TensorBoard and the run log record them as ordinary curves, and
    the generated text goes through the usual text sink as `Sanity/sample_<i>`.
    `SFTPipeline` wires it up from `SFTConfig.sanity_samples`; the samples are pooled
    over every stage and spread evenly, so a multi-stage run probes every corpus it
    trains on.

    A failure of any single sample is written to the console and swallowed; the probe
    must never bring training down. `state_dict()` is not implemented on purpose: the
    baseline of the regression check is a convenience, not training state, and it simply
    restarts after a resume.

    Args:
        samples (Sequence[ProbeSample]): The fixed samples to score.
        tokenizer: Tokenizer used to decode the generated text.
        every_steps (int): Step interval between probes; 0 probes only at the start of
            training.
        max_new_tokens (int): Generation budget per sample.
        temperature (float): Sampling temperature.
        eos_token_id (Optional[int]): End token; None resolves the chat end token of the
            tokenizer (`<|im_end|>`, see `resolve_eos_token_id`), which is where a
            rendered conversation stops.
        sink (Optional[Callable[[str, str, int], None]]): Text sink receiving the reply
            as `Sanity/sample_<i>`.
        generate (bool): Whether to generate at all; False keeps only the loss / accuracy
            part (useful on a slow device).
        warn_ratio (float): Fraction by which the mean loss may grow, relative to the
            first probe, before it is reported.
        ignore_index (int): Label value marking an unsupervised position.
    '''

    def __init__(
        self,
        samples: Sequence[ProbeSample],
        tokenizer,
        every_steps: int = 200,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
        sink: Optional[Callable[[str, str, int], None]] = None,
        generate: bool = True,
        warn_ratio: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        '''
        Configure the probe.

        Raises:
            ValueError: If no sample was given.
        '''
        if not samples:
            raise ValueError('SanityProbe needs at least one probe sample')
        self.samples: List[ProbeSample] = list(samples)
        self.tokenizer = tokenizer
        self.every_steps = int(every_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        if eos_token_id is None:
            eos_token_id = resolve_eos_token_id(tokenizer)
        self.eos_token_id = eos_token_id
        self.sink = sink
        self.generate = bool(generate)
        self.warn_ratio = float(warn_ratio)
        self.ignore_index = int(ignore_index)
        #: Mean loss of the first successful probe; the regression reference.
        self.baseline: Optional[float] = None

    @torch.no_grad()
    def _measure(self, model, sample: ProbeSample, device) -> Tuple[float, float, int]:
        '''
        Teacher-forced loss and next-token accuracy of one sample.

        Only the supervised positions count, exactly as in training: the loss is the
        mean over the model turns, not diluted by the padding or the prompts.

        Args:
            model: Model to score with.
            sample (ProbeSample): The sample.
            device: Device to run on.

        Returns:
            Tuple[float, float, int]: `(loss, accuracy, supervised token count)`; the
            two numbers are NaN when the sample has no supervised token.
        '''
        input_ids = sample.input_ids.unsqueeze(0).to(device)
        labels = sample.labels.unsqueeze(0).to(device)
        batch = {
            'input_ids': input_ids,
            'attention_mask': sample.attention_mask.unsqueeze(0).to(device),
        }
        model_kwargs = {
            key: move_to_device(value, device)
            for key, value in select_model_kwargs(model, batch).items()
        }
        output = model(**model_kwargs)
        logits = getattr(output, 'logits', output)

        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        mask = shift_labels != self.ignore_index
        count = int(mask.sum().item())
        if count == 0:
            return float('nan'), float('nan'), 0

        selected = shift_logits[mask]
        target = shift_labels[mask]
        loss = float(F.cross_entropy(selected, target).item())
        accuracy = float((selected.argmax(dim=-1) == target).float().mean().item())
        return loss, accuracy, count

    @torch.no_grad()
    def _generate(self, model, sample: ProbeSample, device) -> Tuple[str, List[int]]:
        '''
        Complete the prompt of one sample.

        Args:
            model: Model to generate with.
            sample (ProbeSample): The sample whose `prompt_ids` is continued.
            device: Device to run on.

        Returns:
            Tuple[str, List[int]]: The generated text and its token ids.

        Raises:
            RuntimeError: If the model has no sampler-driven `generate` (mini-motif).
        '''
        kwargs = build_generate_kwargs(
            model, self.max_new_tokens, self.temperature, self.eos_token_id
        )
        if 'sampler' not in kwargs:
            raise RuntimeError(
                'the sanity probe drives generation through `TextGeneration`, which '
                'needs a mini-motif `generate(input_ids, ..., sampler)`'
            )
        generation = TextGeneration(
            model,
            self.tokenizer,
            sample.prompt_ids.unsqueeze(0).to(device),
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            sampler=kwargs['sampler'],
            device=device,
            skip_special_tokens=False,
        )
        generation.run()
        return generation.text, [int(token) for token in generation.generated[0].tolist()]

    def _report(self, pipeline) -> Dict[str, float]:
        '''
        Score and generate from every sample, report the result, return the metrics.

        Args:
            pipeline: The training pipeline.

        Returns:
            Dict[str, float]: The `sanity/*` scalars (empty when the pipeline has no
            tokenizer).
        '''
        tokenizer = getattr(pipeline, 'tokenizer', None)
        if tokenizer is None:
            return {}
        model = getattr(pipeline, 'raw_model', None)
        if model is None:
            model = pipeline.model
        device = pipeline.device
        step = pipeline.state.global_step

        was_training = model.training
        model.eval()
        losses: List[float] = []
        counts: List[int] = []
        accuracies: List[float] = []
        warnings: List[str] = []
        try:
            for index, sample in enumerate(self.samples):
                try:
                    loss, accuracy, count = self._measure(model, sample, device)
                except Exception as exc:
                    _write(f'[!] sanity probe could not score {sample.name}: {exc}')
                    continue
                if count > 0:
                    losses.append(loss)
                    counts.append(count)
                    accuracies.append(accuracy)

                text = ''
                ids: List[int] = []
                if self.generate:
                    try:
                        text, ids = self._generate(model, sample, device)
                    except Exception as exc:
                        _write(f'[!] sanity probe could not generate {sample.name}: {exc}')
                    else:
                        if self.sink is not None:
                            self.sink(f'Sanity/sample_{index}', text, step)

                line = (
                    f'[sanity@{step}] {sample.name} loss={loss:.4f} '
                    f'ppl={_perplexity(loss):.2f} acc={accuracy:.3f} supervised={count}'
                )
                if self.generate:
                    line += f' -> {_short(text)!r}'
                _write(line)

                if not math.isfinite(loss):
                    warnings.append(f'{sample.name}: the loss is not finite ({loss})')
                if count == 0:
                    warnings.append(f'{sample.name}: no supervised token at all')
                if self.generate:
                    if not text.strip():
                        warnings.append(f'{sample.name}: the reply is empty')
                    elif not text.replace('<|im_end|>', '').strip():
                        warnings.append(
                            f'{sample.name}: the model stopped right away '
                            f'({_short(text, 40)!r})'
                        )
                    elif _repeats(ids):
                        warnings.append(
                            f'{sample.name}: the reply looks like a repetition loop '
                            f'({_short(text, 40)!r})'
                        )
        finally:
            if was_training:
                model.train()

        metrics: Dict[str, float] = {}
        if losses:
            mean_loss = sum(losses) / len(losses)
            total = sum(counts)
            metrics['sanity/loss'] = mean_loss
            metrics['sanity/ppl'] = _perplexity(mean_loss)
            metrics['sanity/acc'] = sum(a * c for a, c in zip(accuracies, counts)) / total
            metrics['sanity/supervised'] = float(total)
            if self.baseline is None:
                self.baseline = mean_loss
            elif (
                math.isfinite(self.baseline)
                and self.baseline > 0
                and mean_loss > self.baseline * self.warn_ratio
            ):
                warnings.append(
                    f'the mean loss grew {mean_loss / self.baseline:.1f}x since the first '
                    f'probe ({self.baseline:.4f} -> {mean_loss:.4f})'
                )
        metrics['sanity/warnings'] = float(len(warnings))
        for warning in warnings:
            _write(f'[!] sanity@{step}: {warning}')
        return metrics

    def _safe_report(self, pipeline) -> Dict[str, float]:
        '''
        `_report` that never raises: a sanity check may not bring training down.

        Args:
            pipeline: The training pipeline.

        Returns:
            Dict[str, float]: The `sanity/*` scalars, or an empty dict when the probe
            itself failed.
        '''
        try:
            return self._report(pipeline)
        except Exception as exc:
            _write(f'[!] sanity probe failed: {exc}')
            return {}

    def on_train_start(self, pipeline) -> None:
        '''
        Run the probe once before training starts.

        Args:
            pipeline: The training pipeline.
        '''
        self._safe_report(pipeline)

    def on_step_end(self, pipeline, step_metrics: Dict[str, float]) -> None:
        '''
        Run the probe every `every_steps` steps and publish its scalars.

        The metrics are merged into `step_metrics` (the dict every later callback
        receives), which is how they reach TensorBoard and the run log; `SFTPipeline`
        therefore attaches this callback before the monitoring plug-ins.

        Args:
            pipeline: The training pipeline.
            step_metrics (Dict[str, float]): Metrics of this step.
        '''
        if self.every_steps and pipeline.state.global_step % self.every_steps == 0:
            step_metrics.update(self._safe_report(pipeline))
