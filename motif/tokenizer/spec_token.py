from typing import Optional, Union, List


def token(name: Union[str, List[str]], range_num: Optional[int] = None) -> Union[str, List[str]]:
    '''
    Build special-token text of the form `<|name|>`.

    Args:
        name (Union[str, List[str]]): Token name, or a list of names (each is wrapped
            separately).
        range_num (Optional[int]): When given, an indexed series
            `<|name_0|>, <|name_1|>, ...` of that length is returned instead.

    Returns:
        Union[str, List[str]]: The token text, or the list of token texts.
    '''
    if isinstance(name, list):
        return [token(content) for content in name]
    if range_num is None:
        return f'<|{name}|>'
    return [f'<|{name}_{idx}|>' for idx in range(range_num)]


BASIC = token([
    'pad', 'unk', 'sep', 'mask', 'bos', 'eos', 'end_of_text', 'safe_escape'
])

IM_CHAT = token([
    'im_start', 'im_end'
])

ROLE = token([
    'system', 'human', 'tool_response', 'environment',
    'model'
])

COT = token([
    'thought_start',
    'effort_low', 'effort_high', 'effort_max',
    'thought_end'
])

TOOL_STRUCT = token([
    'tool_declare_start', 'tool_declare_end',
    'tool_call_start', 'tool_call_end', 'tool_name_divider'
])

FIM = token([
    'fim_prefix', 'fim_suffix', 'fim_middle'
])

MULTIMODAL = token([
    'modality_image_start', 'modality_image_pad', 'modality_image_end',
    'modality_audio_start', 'modality_audio_pad', 'modality_audio_end',
    'modality_video_start', 'modality_video_pad', 'modality_video_end',
    'timestamp_pad', 'timestamp_frame_divider', 'frame_pad', 'frame_sep'
])

PLAN = token([
    'plan_start', 'plan_sep', 'plan_end'
])

ACTION = token([
    'action_start', 'action_pad', 'action_end',
    'diffusion_start', 'diffusion_pad', 'diffusion_end'
])

GROUNDING = token([
    'bbox_start', 'bbox_end'
])

OBJECT_REF = token([
    'object_ref_start', 'object_ref_end'
])

SPEC_TOKEN = BASIC + IM_CHAT + ROLE + COT + TOOL_STRUCT + FIM + MULTIMODAL + PLAN + ACTION + GROUNDING + OBJECT_REF
SPEC_TOKEN += token('unused', 256 - len(SPEC_TOKEN))
