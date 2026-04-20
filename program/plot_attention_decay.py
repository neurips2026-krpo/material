import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


LABEL_RENDER_MODE = "pinyin"  # "pinyin" for English-paper friendly labels, "chinese" if server fonts are configured

PATIENT_SYMPTOM_TEXT = "体形肥胖，两胁痛，胸闷背痛，胃中嘈杂，脘腹胀满，进食后尤甚，食欲亢进，喜食肥甘，口苦咽干，大便偏干，舌黯红，苔腻，脉沉实"

PREFERRED_SYMPTOM_LABELS = [
    "肥", "胖", "胁", "痛", "胸", "闷", "胃", "嘈", "腹", "胀",
    "口", "苦", "咽", "干", "便", "舌", "苔", "脉", "腻", "沉", "实",
]


PINYIN_FALLBACK_MAP = {
    '湿': 'shi', '热': 're', '中': 'zhong', '阻': 'zu', '汗': 'han', '出': 'chu',
    '恶': 'e', '寒': 'han', '胸': 'xiong', '闷': 'men', '腹': 'fu', '胀': 'zhang',
    '痛': 'tong', '口': 'kou', '苦': 'ku', '咽': 'yan', '干': 'gan', '便': 'bian',
    '大': 'da', '舌': 'she', '苔': 'tai', '腻': 'ni', '脉': 'mai', '肥': 'fei',
    '胖': 'pang', '胁': 'xie', '胃': 'wei', '嘈': 'cao', '食': 'shi', '欲': 'yu',
    '亢': 'kang', '进': 'jin', '喜': 'xi', '甘': 'gan', '红': 'hong', '黯': 'an',
    '实': 'shi', '沉': 'chen', '结': 'jie', '石': 'shi', '胆': 'dan', '囊': 'nang',
}


def clean_token_text(token_text):
    return token_text.replace('Ġ', '').replace(' ', '').replace('Ċ', '\\n').strip()


def decode_token_for_display(tokenizer, token_id):
    token_text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return clean_token_text(token_text)


def decode_token_list_for_display(tokenizer, token_ids):
    return [decode_token_for_display(tokenizer, token_id) for token_id in token_ids]


def find_subsequence(haystack, needle):
    if not needle or len(needle) > len(haystack):
        return -1

    max_start = len(haystack) - len(needle) + 1
    for start_idx in range(max_start):
        if haystack[start_idx:start_idx + len(needle)] == needle:
            return start_idx
    return -1


def locate_prompt_token_span(tokenizer, prompt_text, target_text, full_input_ids):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    start_idx = find_subsequence(prompt_ids, target_ids)
    if start_idx == -1:
        raise ValueError("Could not locate the patient symptom span inside the prompt tokens.")

    special_token_offset = len(full_input_ids) - len(prompt_ids)
    span_start = special_token_offset + start_idx
    span_end = span_start + len(target_ids)
    return span_start, span_end


def romanize_token_text(token_text):
    pieces = []
    for char in token_text:
        if char == '\\n':
            pieces.append(' / ')
        elif char.isascii():
            pieces.append(char)
        else:
            pieces.append(PINYIN_FALLBACK_MAP.get(char, f'u{ord(char):x}'))

    romanized = ''.join(pieces)
    romanized = ' '.join(romanized.replace('_', ' ').split())
    return romanized or 'token'


def format_token_label(token_text):
    if LABEL_RENDER_MODE == "pinyin":
        return romanize_token_text(token_text)
    return token_text


def select_sparse_tick_labels(labels, target_count=6):
    ignored_tokens = {'', '\\n', '，', '。', '：', '；', '、', '【', '】', '"', "'", '(', ')', '[', ']'}
    informative_tokens = [
        (idx, label) for idx, label in enumerate(labels)
        if label not in ignored_tokens
    ]

    if not informative_tokens:
        return [], []

    if len(informative_tokens) <= target_count:
        selected = informative_tokens
    else:
        sample_indices = np.linspace(0, len(informative_tokens) - 1, num=target_count, dtype=int)
        selected = []
        seen_positions = set()
        for sample_idx in sample_indices:
            position, label = informative_tokens[sample_idx]
            if position not in seen_positions:
                selected.append((position, label))
                seen_positions.add(position)

    tick_positions = [position + 0.5 for position, _ in selected]
    tick_labels = [f'[{format_token_label(label)}]' for _, label in selected]
    return tick_positions, tick_labels


def select_preferred_tick_labels(labels, preferred_labels, target_count=6):
    label_positions = []
    used_positions = set()

    for preferred_label in preferred_labels:
        for idx, label in enumerate(labels):
            if idx in used_positions:
                continue
            if preferred_label in label:
                label_positions.append((idx, preferred_label))
                used_positions.add(idx)
                break
        if len(label_positions) >= target_count:
            break

    if len(label_positions) < target_count:
        fallback_positions, fallback_labels = select_sparse_tick_labels(labels, target_count=target_count)
        for position, fallback_label in zip(fallback_positions, fallback_labels):
            original_idx = int(position - 0.5)
            if original_idx in used_positions:
                continue
            label_positions.append((original_idx, labels[original_idx]))
            used_positions.add(original_idx)
            if len(label_positions) >= target_count:
                break

    tick_positions = [idx + 0.5 for idx, _ in label_positions]
    tick_labels = [f'[{format_token_label(label)}]' for _, label in label_positions]
    return tick_positions, tick_labels


def build_generation_stage_ticks(gen_len, window_size):
    if gen_len < window_size * 3:
        candidate_steps = [0, max(gen_len // 2, 0), max(gen_len - 1, 0)]
        stage_names = ['Early', 'Mid', 'Late']
        tick_positions = []
        tick_labels = []
        seen_steps = set()

        for stage_name, step_idx in zip(stage_names, candidate_steps):
            if step_idx in seen_steps:
                continue
            tick_positions.append(step_idx + 0.5)
            tick_labels.append(f'{stage_name}\\nStep {step_idx + 1}')
            seen_steps.add(step_idx)

        return tick_positions, tick_labels

    idx_early = (0, window_size)
    idx_mid = (gen_len // 2 - window_size // 2, gen_len // 2 + window_size // 2)
    idx_late = (gen_len - window_size, gen_len)

    stage_ranges = [
        ('Early', idx_early, window_size / 2),
        ('Mid', idx_mid, window_size + 2 + window_size / 2),
        ('Late', idx_late, window_size * 2 + 4 + window_size / 2),
    ]

    tick_positions = []
    tick_labels = []
    for stage_name, (start_idx, end_idx), position in stage_ranges:
        tick_positions.append(position)
        tick_labels.append(f'{stage_name}\\nStep {start_idx + 1}-{end_idx}')

    return tick_positions, tick_labels


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "attention_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot attention decay and token-to-token heatmaps for a causal LM prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", ""),
        help="Local model path or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=os.getenv("ATTENTION_PROMPT_FILE", ""),
        help="Optional UTF-8 text file. If omitted, the built-in example prompt is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv("ATTENTION_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "7805")),
        help="Maximum number of generated tokens to inspect.",
    )
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_DIR = Path(ARGS.output_dir).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 基础配置
MODEL_PATH = ARGS.model_path.strip()
if not MODEL_PATH:
    raise ValueError("Please provide --model_path or set MODEL_PATH before running this script.")
PROMPT = """你是资深中医专家，精通三焦辨证。\n\n【辨证任务】执行三焦辨证链式推理，并输出统一结论。\n\n【患者临床表现】\n性别：女性\n年龄：35岁\n临床表现：体形肥胖，两胁痛，胸闷背痛，胃中嘈杂，脘腹胀满，进食后尤甚，食欲亢进，喜食肥甘，口苦咽干，大便偏干，舌黯红，苔腻，脉沉实\n病史：B超示胆囊结石\n\n【症状知识库】\n[\n  \"肥胖：体胖痰多，常见身重困倦。  \\n可见于痰阻心脉证，多因痰浊阻滞心脉所致，故见体胖痰多并伴闷痛、舌苔白腻、脉沉滑。  \\n亦可见于脾气虚证，因脾气虚导致水湿不运、泛溢肌肤而致体胖。\",\n  \"胁痛：胁痛可发生于一或双侧胁肋部，疼痛性质包括胀痛、窜痛、刺痛或隐痛，多拒按，偶有喜按者，常反复发作，且在呼吸、咳嗽或转侧时牵引作痛。  \\n常见于肝气郁结证，多因情志不舒或暴怒气逆所致，其病机为肝脉不畅、气机阻滞，不通则痛。  \\n可见于瘀血阻络证，多由肝郁气滞及血或跌仆损伤致瘀血停留，阻滞胁络，不通则痛。  \\n亦见于湿热蕴结证，因外感湿热或饮食不节生湿蕴热，蕴结肝胆，疏泄不利，气机阻滞而致胁痛。  \\n还见于肝阴不足证，多因精血亏损致肝阴不足，络脉失养，不荣则痛。  \\n此外，胁痛胀满可见于肝火炽盛证；胁肋隐痛可见于肝阴虚证及肝肾阴虚证；胸胁窜痛可见于肝气横逆证，并伴暴躁易怒、脘腹胀痛及呃逆呕恶。\",\n  \"胸痛：胸痛是以胸部疼痛为主要表现的自觉症状，多由病邪壅阻心胸血脉、气血不通所致，常见实证，病邪包括寒、热、痰、瘀，亦可见本虚标实证。\\n\\n胸痛憋闷有压榨感常见于气滞或痰阻；胸痛如刺、夜间加重多见于血瘀阻滞。胸痛连脘腹不可触者为寒热结胸；痛连胁部病在肝胆；痛引左手尺侧为胸痹心痛；痛引肩背伴发热呕恶为肝胆湿热；痛连肩背脉沉紧为寒凝心胸。\\n\\n胸痛伴发热咳嗽、咳则痛甚多见于肺热络伤；伴咳吐脓血痰为肺痈；胸部隐痛、咳嗽无力常见于肺气虚弱或肺热病后期，亦见于肺痨；伴心悸病在心；心胸卒然大痛持续不解，面青肢冷脉微细者为心脉闭阻之真心痛。\\n\\n胸闷常伴见于多种证型：湿郁气机证见胸闷脘痞；实证见胸闷烦躁；上实证见胸闷脘胀；暑湿证见胸闷纳呆；脏腑气闭证见胸闷腹胀；痰瘀结于心脑证见心胸闷痛、绞痛伴头目胀痛、痴呆等；痰瘀结于肺证见胸闷胸痛；湿阻脾阳证见胸闷脘痞恶心；肺热炽盛证见气促胸闷；气血运行迟缓证见胸闷腹胀；毒虫所伤、心气亏虚、胸阳不振、阳虚证、邪热灼津、痰阻气道、胆郁痰扰、痰火扰神、气机不畅及心肺气虚等证均可见胸闷或胸闷气短。\\n\\n其他证型中：邪热壅肺、痰瘀结于肺、温燥、肺热炽盛、心阳虚衰、燥伤肺络、热邪犯肺、痰热阻滞肺络等亦可见胸痛，其病机多与肺失宣降、热壅气滞或痰瘀阻滞相关。\",\n  \"心嘈：胃脘嘈杂不适。  \\n可见于虫居肠道证，因虫体争食水谷、吮吸精微所致。\",\n  \"腹满：腹满主要表现为腹部胀满，可伴有胀痛、硬痛或拒按，时或缓解，常与嗳腐吞酸、呕吐、便闭等症状并见。\\n\\n常见于热结肠胃证，可见腹满拒按、大便干结，其病机为胃肠热盛、津伤肠燥、糟粕内结，燥屎阻遏气机。  \\n亦见于食积胃脘证，多见脘腹胀满疼痛拒按，并见嗳腐吞酸、纳呆厌食，病机为食积胃脘、气机不畅。  \\n太阴病本证可见腹满时痛，伴呕吐、下利、食不下，病因属中焦虚寒、寒湿内阻，病机为中阳不足、运化失职、寒湿内停。  \\n肠热腑实证亦见腹满硬痛、便秘，病机为里热炽盛、腑气不通。  \\n湿阻大肠证可见少腹满、大便不通、苔垢腻。  \\n有形邪热证可见腹满、腹痛、便闭，病机属燥屎结实阻遏气机。  \\n胃热炽盛证亦可见腹胀腹满、大便秘结。\",\n  \"腹胀：腹胀主要表现为腹部胀满，可伴疼痛、纳差、便溏或便秘。常见拒按或食后加重。\\n\\n可见于里实证，多因燥实内结所致，病机为腑气不通，故见腹胀满痛拒按、大便秘结。  \\n可见于肝郁气滞证，多因肝郁气滞、气机不畅所致，故见胸胁或少腹胀闷窜痛。  \\n可见于脾气虚证，多因脾虚运化失司所致，故见纳少腹胀便溏。  \\n可见于胃肠气滞证，多因胃肠气机阻滞所致，故见脘腹胀痛走窜。  \\n可见于湿热侵犯大肠或下焦湿热证，多因湿热壅阻气机所致，故见腹痛腹胀或腰、小腹胀痛。  \\n可见于肝胆湿热证，多因湿热郁阻脾胃所致，故见胁痛厌食腹胀。  \\n可见于肝郁脾虚证，多因肝气犯脾、脾失健运所致，故见胸胁胀痛、腹胀纳呆便溏。  \\n\\n亦可见于脏腑气闭证，病机为传导、气化不行，故见胸闷腹胀、二便不通。  \\n或见于气血运行迟缓、痰湿内停之证，病机渐致气滞血瘀、经络痹阻，故见胸闷腹胀、二便不利。  \\n\\n腹部胀大如囊裹水者，病机多为水邪蓄积于腹腔。  \\n脘腹重坠作胀者，多见于脾气虚衰证，病机为升举无力、内脏失托。  \\n食后胀甚者，可见于胃气亏虚证，病机为胃气失和、受纳腐熟功能减退。  \\n腹胀如鼓者，病机属土不制水，反受其克。  \\n\\n其他如寒湿发黄证可见脘腹胀闷；思伤脾胃证可见腹胀便溏；痞满燥实证可见腹部胀满坚硬。\",\n  \"消谷善饥：消谷善饥常见发热、健忘、大便黑硬而反易、小便正常、舌红苔紫暗或有瘀点、脉数或涩等表现。  \\n常见于胃火炽盛证，多因胃火炽盛、机能亢进所致，因此可见本症。\",\n  \"嗜食：嗜食表现为偏好辛辣刺激、温补之品、肥腻厚味或生冷食物。  \\n可见于脾胃功能失调相关证候，多由饮食失调、过用寒凉或攻伐药物、脾胃素弱、阳气自衰及久病失养等因素所致。其病机多为久蕴化热生火或湿热内生，影响脾胃纳运功能，故见嗜食。\",\n  \"口苦：口苦是自觉口中发苦的症状，常见于多种肝胆系及热性证候。  \\n可见于胆热上犯证，因胆热上炎犯胃所致。  \\n少阳病证及少阳涉病证亦多见口苦，常伴咽干、往来寒热等，其病机为少阳枢机郁热。  \\n肝火炽盛或肝火上炎证亦常见口苦，多因肝火炽盛、耗伤津液，并挟胆热上蒸所致。  \\n湿热类证型如湿热蕴脾、湿热郁蒸、肝胆湿热等，均可见口苦，病机多为湿热郁蒸、胆气上溢或肝胆疏泄障碍、胆汁外溢。  \\n此外，热迫胆气或热蒸胆气上溢亦可导致口苦。\",\n  \"咽干：咽干主要表现为咽喉干燥不适，常与口燥并见。  \\n常见于热盛伤津证，因热盛耗伤津液所致。  \\n可见于阴虚证，多由阴液滋养不足，机体失于润泽而引发。  \\n心肾虚证亦可见口燥咽干，病因包括阴水不足、里虚热或邪热伤阴，病机为阴虚火旺。  \\n津不上承证及阴亏津不上承证均因津液无法上润咽喉而致咽干。  \\n肝火上炎证因肝火耗津、挟胆热上蒸，可见口苦咽干。  \\n阴津亏虚证因津液不能上输润喉，故见口燥咽干。  \\n燥邪伤肺证因燥邪伤肺、失于滋润，可见咽干或口鼻咽干。  \\n肾阴精不足证因肾阴精亏少致唾液减少，可见口燥咽干。  \\n肾阴亏虚证及阴虚内热证均因阴不制阳、虚火内生，可见咽干或口燥咽干。  \\n此外，咽干颧红可见于肾阴亏虚证，伴见潮热盗汗、五心烦热等，病机为虚火内生。  \\n口干咽痛可见于风热上犯咽喉证，因风热灼伤津液所致。\",\n  \"大便干结：大便干结主要表现为粪便干燥、难以排出。  \\n常见于热结肠胃证，多因热邪结滞肠道所致，病机为传导失司，可见腹胀满痛拒按并大便干结。  \\n亦可见于阴虚证或津液不足证，病因多为阴液或津液滋养不足，病机为肠道失润、传导不利，常伴口燥咽干、小便短黄等症状。\",\n  \"少腹急结：少腹急结主要表现为下腹部拘急结聚、硬满不适。  \\n常见于太阳蓄血证，多因邪热与瘀血搏结于下焦所致，其病机为邪热瘀血搏结下焦，因此可见本症。\"\n]\n\n【上焦证辨析目标】\n定位与阶段：胸中之肺与心包，为温病初起多见的病位。证候与卫气营血对应：肺卫受邪属卫分；邪热壅肺属气分；热入心包多涉营分；肺热极致入络致咯血涉血分。推理路径：先判表里（有无恶风寒、汗、苔薄白/黄）；再判病位（咳嗽、咳喘为肺；神志改变为心包）；再判病性（风热或湿热）；再判进退（顺传中焦或逆传心包）。功能失调环节：肺宣发肃降失司、气机闭郁；心包主神明之机窍可被热/痰/瘀所闭。因果链：外感温邪自口鼻上受→犯肺卫或直犯肺→里热壅盛可入心包或循经下传中焦→进一步再及下焦。兼证关系与交错：上中下可并见与交错，需根据时间序列与强弱判主次。证据锚点：症状体征与舌脉映射上述病机，避免仅以发热孤证定类。三焦传变：温病多始于上焦肺卫，常按上→中→下顺传；亦可因邪性、体质与伏邪而越经直中或逆传心包。顺传判据：脏传腑、热势自表向里而有外透，便不闭或有下泄为邪有出路，正气尚可；常见路径为肺卫→阳明胃腑。逆传判据：肺卫未解而热陷心包，属脏传脏，起病急骤，神志异常、四末厥冷、舌质红绛为要证，预后较凶。影响因素：暑热可直犯心包；湿热可直阻中焦；肾精素虚或伏邪温病可始发于下焦或营血分。并行与交错：上中下焦病变可交错重叠，需按起点→主位→兼夹（湿、痰、瘀）→顺逆→虚实流程判定。决策流程：1. 明确起病阶段与病位线索；2. 以表里与腑实/经热为纲判气机趋向；3. 识别顺传/逆传标志与卫气营血层次；4. 标注兼夹因素（痰、湿、瘀）与正虚（津伤、阴伤）；5. 结合时间序列评估预后与风险点。\n【上焦证候选证型】\n邪犯肺卫证, 肺热壅盛证, 湿热阻肺证, 热陷心包证, 湿蒙心包证\n【上焦证辨证规则】\n- 上焦证：定位与阶段：胸中之肺与心包，为温病初起多见的病位。证候与卫气营血对应：肺卫受邪属卫分；邪热壅肺属气分；热入心包多涉营分；肺热极致入络致咯血涉血分。推理路径：先判表里（有无恶风寒、汗、苔薄白/黄）；再判病位（咳嗽、咳喘为肺；神志改变为心包）；再判病性（风热或湿热）；再判进退（顺传中焦或逆传心包）。功能失调环节：肺宣发肃降失司、气机闭郁；心包主神明之机窍可被热/痰/瘀所闭。因果链：外感温邪自口鼻上受→犯肺卫或直犯肺→里热壅盛可入心包或循经下传中焦→进一步再及下焦。兼证关系与交错：上中下可并见与交错，需根据时间序列与强弱判主次。证据锚点：症状体征与舌脉映射上述病机，避免仅以发热孤证定类。三焦传变：温病多始于上焦肺卫，常按上→中→下顺传；亦可因邪性、体质与伏邪而越经直中或逆传心包。顺传判据：脏传腑、热势自表向里而有外透，便不闭或有下泄为邪有出路，正气尚可；常见路径为肺卫→阳明胃腑。逆传判据：肺卫未解而热陷心包，属脏传脏，起病急骤，神志异常、四末厥冷、舌质红绛为要证，预后较凶。影响因素：暑热可直犯心包；湿热可直阻中焦；肾精素虚或伏邪温病可始发于下焦或营血分。并行与交错：上中下焦病变可交错重叠，需按起点→主位→兼夹（湿、痰、瘀）→顺逆→虚实流程判定。决策流程：1. 明确起病阶段与病位线索；2. 以表里与腑实/经热为纲判气机趋向；3. 识别顺传/逆传标志与卫气营血层次；4. 标注兼夹因素（痰、湿、瘀）与正虚（津伤、阴伤）；5. 结合时间序列评估预后与风险点。\n- 上焦证-邪犯肺卫证：起病特征：温邪上受，新感急起，表证与肺系同见。判别锚点：发热微恶风寒、咳嗽、头痛、口微渴，舌边尖红，苔薄白欠润，脉浮数。核心病机：温邪遏郁卫气，肺卫同受，肺宣发失司→咳嗽；卫郁于表→微恶风寒。因果链：外感温邪→郁卫犯肺→肺失宣肃→咳嗽与头痛。鉴别要点：与邪热壅肺相比，本证有表无里；与湿热阻肺相比，苔薄白而不腻、身热不扬不显。层次归属：卫分为主，可向气分转化。\n- 上焦证-肺热壅盛证：证据锚点：身热汗出、咳喘气促、口渴喜冷、苔黄、脉数。核心病机：邪热壅肺，肺气闭郁，津液被热所耗。因果链：肺卫未解或邪直入里→邪热亢盛壅塞肺络→宣降失司→咳喘气促；热盛耗津→汗出、口渴。鉴别要点：与邪犯肺卫相比，无明显恶寒表证；与湿热阻肺相比，热象重、苔黄而不以白腻为主。层次归属：气分为主，可兼见入络出血则涉血分。\n- 上焦证-湿热阻肺证：证据锚点：恶寒发热、身热不扬、胸闷、咳嗽、咽痛、苔白腻、脉濡缓。核心病机：湿郁卫表，湿热阻滞肺气，宣降不利。因果链：湿邪犯表→卫表受遏→气机不展→胸闷；湿热上扰肺系→咳嗽咽痛；湿郁为主→舌苔白腻、身热不扬。鉴别要点：与邪犯肺卫相比，苔由薄白转白腻、身重困倦可加；与肺热壅盛相比，热势不炽，咳喘不甚。层次归属：介于卫与气分之间，偏湿。\n- 上焦证-热陷心包证：证据锚点：身灼热、神昏或谵语、肢厥、舌蹇、舌质红绛。核心病机：热邪内陷包络，机窍被闭，神明被扰；常见夹痰或夹瘀。因果链：上焦热盛不解或逆传→热陷心包→心神被扰与机窍闭阻→神昏谵语、四末厥冷；营血受灼→舌红绛。鉴别要点：与湿蒙心包相比，热势急骤、神志障碍更重、四末厥冷；舌质多红绛而非仅苔腻。层次归属：气分入营分之危重阶段。\n- 上焦证-湿蒙心包证：证据锚点：身热、神识时清时昧、间有谵语、舌苔垢腻、舌质不绛。核心病机：气分湿热酿痰浊，蒙蔽包络，心神被困而非深度闭陷。因果链：湿热久羁中上焦→痰浊酿生→蒙蔽心包→神识昏蒙时清时昧；湿热上泛→苔垢腻。鉴别要点：与热陷心包相比，无明显四肢厥冷，舌质不绛、昏蒙较轻。层次归属：气分湿热为主，可向营分转化。\n\n【中焦证辨析目标】\n定位与阶段：胃、脾、肠等中焦；多见于温病中期或极期。总体特征：邪盛而正未大伤，邪正斗争剧烈，治当助正祛邪则易得出路；若热盛腑实或湿热互结，则津液耗伤、真阴渐损。卫气营血对应：阳明胃经热盛属气分经证；肠腑热结属气分腑证；热逼血妄行见斑疹或肠络受伤便血则可涉血分。推理路径：先判腑实与否（便秘、腹满痛、苔黄黑燥、脉沉实）；再判湿热权重（白腻/黄腻与身重困倦）；再判气机（升清降浊失司与否）；结合时间序列评估传变趋势。功能失调环节：阳明燥热耗津、脾失健运、肠道传导失司。兼证关系：可与上焦肺热/心包证并见，或向下焦阴伤演变。\n【中焦证候选证型】\n阳明热炽证, 阳明热结证, 湿邪困脾证, 湿热中阻证, 湿热积滞搏结肠腑证, 湿阻大肠证\n【中焦证辨证规则】\n- 中焦证：定位与阶段：胃、脾、肠等中焦；多见于温病中期或极期。总体特征：邪盛而正未大伤，邪正斗争剧烈，治当助正祛邪则易得出路；若热盛腑实或湿热互结，则津液耗伤、真阴渐损。卫气营血对应：阳明胃经热盛属气分经证；肠腑热结属气分腑证；热逼血妄行见斑疹或肠络受伤便血则可涉血分。推理路径：先判腑实与否（便秘、腹满痛、苔黄黑燥、脉沉实）；再判湿热权重（白腻/黄腻与身重困倦）；再判气机（升清降浊失司与否）；结合时间序列评估传变趋势。功能失调环节：阳明燥热耗津、脾失健运、肠道传导失司。兼证关系：可与上焦肺热/心包证并见，或向下焦阴伤演变。\n- 中焦证-阳明热炽证：证据锚点：壮热、大汗出、心烦、面赤、口渴引饮、苔黄或微燥、脉洪大而数。核心病机：热入阳明，里热蒸迫，内外弥漫之散漫浮热。因果链：邪热盛于阳明→蒸津外泄→壮热汗多、口渴多饮；热扰心神→心烦；热上蒸→面赤。鉴别要点：与阳明热结相比，不以大便秘结与腹硬满为主；与肺热壅盛相比，系统性里热外蒸明显而非局限肺系。层次归属：气分经证。\n- 中焦证-阳明热结证：证据锚点：日晡潮热或谵语、大便秘结或热结旁流、腹部硬满疼痛、舌苔黄黑而燥、脉沉实有力。核心病机：里热与燥屎相结，津伤，肠道传导失司。因果链：热结肠腑→津枯腑实→传导不利→便秘或旁流；里热扰神→谵语；腑实内结→腹硬满痛。鉴别要点：与湿热积滞搏结肠腑相比，此证以燥热与便秘为主，苔黄黑燥；彼证便溏如败酱为要。变证提示：邪热损伤肠络蓄血者可见身热夜甚、神志如狂、大便色黑。层次归属：气分腑证。\n- 中焦证-湿邪困脾证：证据锚点：身热不扬、胸脘痞满、泛恶欲呕、身重肢倦、苔白腻、脉濡缓。核心病机：湿重热轻，困阻中州，升运失司，胃失和降。因果链：湿郁中焦→脾运不及→胸脘痞满、身重困倦；胃气上逆→泛恶欲呕；湿为主→舌苔白腻、热象不著。鉴别要点：与湿热中阻相比，热势较轻、苔以白腻为主；与阳明热炽相比，无壮热汗出与洪数脉。层次归属：气分偏湿。\n- 中焦证-湿热中阻证：证据锚点：高热持续（汗出不解）、烦躁不安、脘腹痛满、恶心欲呕、舌苔黄腻或黄浊。核心病机：湿与热互结于中焦，升清降浊受阻。因果链：湿热互蒸→热势不衰→高热不解；中焦壅滞→脘腹痛满；胃失和降→恶心呕吐；苔黄腻/黄浊为湿热互结之征。鉴别要点：与阳明热炽相比，湿象更著（苔黄腻、腹痞）；与阳明热结相比，不以便秘燥屎为主。层次归属：气分湿热。\n- 中焦证-湿热积滞搏结肠腑证：证据锚点：身热、烦躁、胸脘痞满、腹痛不食、大便溏垢如败酱、便下不爽、舌赤、苔黄腻或黄浊、脉滑数。核心病机：肠腑湿热与糟粕相搏，气机不通，传导失司。因果链：湿热熏蒸肠腑→与积滞相搏→肠腑不通→腹痛、便下不爽；湿热蕴结→舌苔黄腻/黄浊、脉滑数。鉴别要点：与阳明热结之燥热便秘相别；与湿阻大肠相比，此证仍多有下泄但不爽。层次归属：气分湿热搏结。\n- 中焦证-湿阻大肠证：证据锚点：大便不通、神识如蒙、少腹硬满、苔垢腻、脉濡。核心病机：湿浊闭阻肠道，上蒙清窍、下闭浊道，传导失职。因果链：湿浊壅滞→腑气不通→少腹硬满与便闭；浊湿上蒙→神识如蒙；苔垢腻、脉濡为湿盛之征。鉴别要点：与阳明热结相比，此证湿盛而非燥实，脉象濡而不沉实。层次归属：气分湿浊闭阻。\n\n【下焦证辨析目标】\n定位与阶段：肝肾为主，多见于温病后期，常呈邪少虚多。与血分动血之别：本证以真阴、精血耗伤为纲，动血妄行者多属热盛迫血，证性偏实或虚实夹杂。卫气营血对应：肾阴耗损与虚风内动多涉营血/血分。推理路径：先判阴精亏损程度（口咽燥、五心烦热、舌绛干枯、脉虚）；再判是否生内风（手指蠕动、瘛疭、肢厥）；并评估兼夹热/湿与及其来源（上焦/中焦传下）。功能失调环节：肾藏精与肝藏血失养→筋脉失濡，水不涵木，心神失所。风险提示：阴竭阳脱为重危节点。\n【下焦证候选证型】\n肾精耗损证, 虚风内动证\n【下焦证辨证规则】\n- 下焦证：定位与阶段：肝肾为主，多见于温病后期，常呈邪少虚多。与血分动血之别：本证以真阴、精血耗伤为纲，动血妄行者多属热盛迫血，证性偏实或虚实夹杂。卫气营血对应：肾阴耗损与虚风内动多涉营血/血分。推理路径：先判阴精亏损程度（口咽燥、五心烦热、舌绛干枯、脉虚）；再判是否生内风（手指蠕动、瘛疭、肢厥）；并评估兼夹热/湿与及其来源（上焦/中焦传下）。功能失调环节：肾藏精与肝藏血失养→筋脉失濡，水不涵木，心神失所。风险提示：阴竭阳脱为重危节点。\n- 下焦证-肾精耗损证：证据锚点：低热、神惫萎顿、消瘦无力、口燥咽干、耳聋、手足心热甚于手足背、舌绛不鲜干枯而痿、脉虚。核心病机：邪热深入下焦，灼耗肾阴与肾精，形神失养。因果链：温邪久羁或下传→真阴受损→津亏热伏→口咽燥、五心烦热；精不足→形体消瘦、神惫。鉴别要点：与中焦腑实证不同，不以腹满痛与便秘为主；与虚风内动相比，本证以内伤为本，未出现明显抽搐蠕动。层次归属：血分（真阴受损）。\n- 下焦证-虚风内动证：证据锚点：神倦肢厥、耳聋、五心烦热、心中憺憺大动、手指蠕动或瘛疭、舌干绛而痿、脉虚弱。核心病机：肾精虚损，肝木失养，水不涵木，内风自生；阴不内守，心神不宁。因果链：真阴久亏→肝肾失濡→筋脉失养→手指蠕动/瘛疭；肾水不足→心火无济→心中悸动不安；阴虚生内热→五心烦热、舌干绛。鉴别要点：在本证之上可承接肾精耗损，不同于热陷心包的急骤神昏与厥冷。层次归属：血分阴虚风动。\n\n请依据【症状知识库】与【辨证规则】，围绕【患者临床表现】进行链式推理，并遵循以下中文说明输出：\n\n1. 输出必须以 <reasoning> 开始、以 </reasoning> 结束。\n\n2. 在 <reasoning> 内部，只允许出现 <analysis>...</analysis> 段与 action 行，且 action 名称与顺序必须与示例完全一致。\n\n3. <analysis> 与 action 必须严格交替，结构示例：\n   <reasoning>\n   <analysis>...</analysis>[要求：在<analysis>块中,列出患者核心症状、诱因与【症状知识库】中的关键锚点，明确可支撑三焦判定的证据链，不能输出本prompt内容。]\n   action1:[症状知识锚定]: [围绕主症与并见症，输出支持/反对/待查的证据，引用知识库原句或要点，保持条理]\n   <analysis>我先回顾三焦辨证包含有上焦证、中焦证、下焦证的辨证类型，因此需要逐个进行辨证。</analysis>\n   action2:[总体计划]: {\"分析结果\":\"下面依次进行上焦证、中焦证、下焦证的辨证推理。\",\"下一步骤\":\"继续进行上焦证辨证\"}\n   <analysis>...</analysis>[要求：在<analysis>块中,沿候选属性顺序回溯【辨证规则】，逐条说明与患者症状的匹配度及理由，不能输出本prompt内容。]\n   action3:[上焦证判定]: {\"分析结果\":\"\",\"简短原因\":\"\",\"下一步骤\":\"继续进行中焦证辨析。\"}\n   <analysis>...</analysis>[要求：在<analysis>块中,沿候选属性顺序回溯【辨证规则】，逐条说明与患者症状的匹配度及理由，不能输出本prompt内容。]\n   action4:[中焦证判定]: {\"分析结果\":\"\",\"简短原因\":\"\",\"下一步骤\":\"继续进行下焦证辨析。\"}\n   <analysis>...</analysis>[要求：在<analysis>块中,沿候选属性顺序回溯【辨证规则】，逐条说明与患者症状的匹配度及理由，不能输出本prompt内容。]\n   action5:[下焦证判定]: {\"分析结果\":\"\",\"简短原因\":\"\",\"下一步骤\":\"进入三焦综合判定。\"}\n   <analysis>...</analysis>[要求：在<analysis>块中,在引用各维度 action 结论的基础上，说明总体三焦归属的逻辑、兼夹情况及证据充足度]\n   action6:[三焦判定结论]: {\"辨证结果\":\"\",\"子类\":\"\",\"简短原因\":\"\",\"下一步骤\":\"推理完毕。\"}\n   </reasoning>\n\n4. 从 action2 开始（含 action2），action 的‘结果内容’必须是单行 JSON 对象，不允许额外解释文本、不允许多行、不允许代码块。\n\n5. action2 的 JSON 必须且只能包含 2 个键：分析结果、下一步骤。\n\n6. action3 及之后（含三焦判定结论）的 JSON 必须且只能包含 3 个键：辨证结果、简短原因、下一步骤。注意：三焦判定结论中包含 辨证结果、子类、简短原因、下一步骤。\n\n7. 辨证结果必须使用对应【候选证型】的原名；证据不足需输出‘无’并在简短原因中点出矛盾点。\n\n8. 简短原因必须≤50字，引用患者信息或症状知识库中的关键证据（自然语言一句话）。\n\n9. 在 </reasoning> 之后追加一行：‘三焦辨证推理分析总结：’，格式为：{\"三焦辨证证型\":\"\",\"综合结论\":\"≤40字综合结论（需包含关键证据或矛盾点）\"}/no_think
""" 

if ARGS.prompt_file:
    PROMPT = Path(ARGS.prompt_file).expanduser().read_text(encoding="utf-8")

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager" 
)
model.eval()
last_layer_index = getattr(model.config, "num_hidden_layers", None)
if last_layer_index is not None:
    last_layer_index -= 1
num_attention_heads = getattr(model.config, "num_attention_heads", None)

# 3. 准备输入数据
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
prompt_length = inputs.input_ids.shape[1] # 获取原始 Prompt 的 Token 数量
print(f"原始 Prompt 长度: {prompt_length} tokens")
# --- 依然需要关闭 Flash Attention 以防底层拦截 ---
model.config.use_flash_attn = False
model.config.use_flash_attention_2 = False

print("开始执行 Prefill (预填充) 阶段，不提取注意力以节省显存...")
# 1. 第一步：处理 7805 个字的超长 Prompt
with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,          # 必须开启 KV Cache
        output_attentions=False  # 【核心救命参数】：禁止输出 7800x7800 的核弹级矩阵！
    )
    
    # 获取 KV Cache 供后续生成使用
    past_key_values = outputs.past_key_values
    # 预测出第一个生成的 Token
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

print("开始单步循环生成，并实时提取小巧的注意力矩阵...")
attention_decay_values = []
generated_token_ids = [next_token_id.item()]

MAX_NEW_TOKENS = ARGS.max_new_tokens
heatmap_vectors = []
# 2. 第二步：手动写 For 循环，逐个字生成
for step in range(MAX_NEW_TOKENS):
    with torch.no_grad():
        # 此时输入模型的是长度仅为 1 的 next_token_id！
        outputs = model(
            input_ids=next_token_id,
            past_key_values=past_key_values, # 传入历史记忆
            use_cache=True,
            output_attentions=True           # 【此时开启安全】：因为 Q 长度为 1，矩阵极小！
        )
    
    # 更新 KV Cache 和预测下一个字
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated_token_ids.append(next_token_id.item())
    
    # 提取 Attention (因为输入 Q 长度为 1，直接取 outputs.attentions)
    if outputs.attentions is not None:
        # 取最后一层: shape -> (batch=1, num_heads, q_len=1, kv_len)
        last_layer_attn = outputs.attentions[-1]
        
        # 步骤 A: 对所有 Attention Heads 求平均 -> (1, 1, kv_len)
        avg_heads_attn = last_layer_attn.mean(dim=1)
        
        # 步骤 B: 提取当前这 1 个 Token 对【原始 Prompt (前7805个位置)】的注意力
        # 第二维度写 0，因为 q_len = 1，就只有索引 0
        attn_to_prompt = avg_heads_attn[0, 0, :prompt_length]
        
        # 步骤 C: 求和保存
        attention_decay_values.append(attn_to_prompt.sum().item())
        # 把它转成 float16 放在 CPU 上，防止内存爆炸
        heatmap_vectors.append(attn_to_prompt.cpu().to(torch.float16).numpy())
    
    # 如果生成了结束符 (EOS)，提前停止
    if next_token_id.item() == tokenizer.eos_token_id:
        print(f"检测到结束符，提前停止于第 {step+1} 步。")
        break
        
    # 每 50 步打印一次进度并清理显存碎片
    if (step + 1) % 50 == 0:
        print(f"已生成 {step + 1} 个 token...")
        # 及时清理临时变量防止显存泄露
        del last_layer_attn
        del avg_heads_attn
        del outputs

# 打印一下生成的文本，看看模型说了啥
generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
print("\n========== 模型生成的回复 ==========")
print(generated_text[:200] + "......(省略)")
print("\n")
print(generated_text[-200:]) # 打印结尾部分，看看有没有合理结束
print("==================================\n")

# 6. 画图部分 
print("正在绘制注意力衰减曲线...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
x_axis = range(1, len(attention_decay_values) + 1)
sns.lineplot(x=x_axis, y=attention_decay_values, color="b", linewidth=2)
plt.title("Attention Decay Curve (Reasoning Tokens to Original Prompt)", fontsize=16)
plt.xlabel("Reasoning Steps (Generated Token Index)", fontsize=14)
plt.ylabel("Sum of Attention Weights to Prompt", fontsize=14)
plt.axhline(y=max(attention_decay_values)*0.5, color='r', linestyle='--', label="50% Attention Threshold")
plt.legend()
curve_path = OUTPUT_DIR / "attention_decay_curve.png"
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {curve_path}")



print("正在绘制 Token-to-Token 经典热力图...")

# ==========================================
# 1. 解决 Matplotlib 中文显示问题 
# ==========================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 矩阵转置与 Y 轴 (Prompt症状) 截取
# ==========================================
# 原始 heatmap_vectors 形状: [生成的Token数, 原始Prompt总长度]
# 转置后: [原始Prompt总长度, 生成的Token数]
full_matrix_T = np.vstack(heatmap_vectors).T

y_start_idx, y_end_idx = locate_prompt_token_span(
    tokenizer,
    PROMPT,
    PATIENT_SYMPTOM_TEXT,
    inputs.input_ids[0].tolist(),
)

# 截取 Y 轴数据
y_matrix = full_matrix_T[y_start_idx:y_end_idx, :]

# 获取 Y 轴对应的中文 Token 文本
prompt_token_ids = inputs.input_ids[0][y_start_idx:y_end_idx]
y_tokens_raw = decode_token_list_for_display(tokenizer, prompt_token_ids)
# 清理 Qwen Tokenizer 可能带有的特殊前缀 (如 Ġ,   等)
y_labels = [clean_token_text(token_text) for token_text in y_tokens_raw]
y_tick_positions, y_tick_labels = select_preferred_tick_labels(
    y_labels,
    preferred_labels=PREFERRED_SYMPTOM_LABELS,
    target_count=6,
)

# ==========================================
# 3. X 轴 (生成的推理链) 分段截取 (前、中、末段)
# ==========================================
gen_len = len(generated_token_ids)
window_size = 40 # 每段截取 40 个 Token

# 如果生成总长度太短，就直接画全图；如果够长，则分三段
if gen_len < window_size * 3:
    final_matrix = y_matrix
    x_token_ids = generated_token_ids
else:
    # 提取前段 (刚开始推理)、中段 (推理一半)、末段 (即将结束)
    idx_early = (0, window_size)
    idx_mid = (gen_len // 2 - window_size // 2, gen_len // 2 + window_size // 2)
    idx_late = (gen_len - window_size, gen_len)
    
    # 将这三段数据在 X 轴方向拼合
    mat_early = y_matrix[:, idx_early[0]:idx_early[1]]
    mat_mid = y_matrix[:, idx_mid[0]:idx_mid[1]]
    mat_late = y_matrix[:, idx_late[0]:idx_late[1]]
    
    # 加入一列 np.nan 作为视觉上的“分割线”
    separator = np.full((y_matrix.shape[0], 2), np.nan)
    
    final_matrix = np.hstack([mat_early, separator, mat_mid, separator, mat_late])
    
    # 拼合对应的 X 轴 Token 文本
    x_token_ids = (generated_token_ids[idx_early[0]:idx_early[1]] + 
                   [0, 0] + # 对应分割线
                   generated_token_ids[idx_mid[0]:idx_mid[1]] + 
                   [0, 0] + # 对应分割线
                   generated_token_ids[idx_late[0]:idx_late[1]])

# 解码 X 轴的 Token 文本
x_tokens_raw = decode_token_list_for_display(tokenizer, x_token_ids)
x_labels = []
for i, t in enumerate(x_tokens_raw):
    if x_token_ids[i] == 0:
        x_labels.append("...") # 分割线标识
    else:
        # 清理特殊符号
        clean_t = clean_token_text(t)
        x_labels.append(clean_t)

x_tick_positions, x_tick_labels = build_generation_stage_ticks(gen_len, window_size)

# ==========================================
# 4. 绘制经典 Seaborn 热力图
# ==========================================
fig, ax = plt.subplots(figsize=(20, 10))
dynamic_vmax = np.nanpercentile(final_matrix, 98) 
ax = sns.heatmap(
    final_matrix, 
    cmap='Blues',      
    vmax=dynamic_vmax,
    vmin=0,
    xticklabels=False,  
    yticklabels=False,
    linewidths=0.5,        
    linecolor='lightgray',
    cbar_kws={'label': 'Attention Weight'},
    ax=ax,
)

ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=11)
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=12)

ax.set_title("Token-to-Token Attention Matrix (Last-Layer Mean Attention)", fontsize=18, pad=20)
ax.set_xlabel("Generated Reasoning Tokens (Time \u2192)", fontsize=14)
ax.set_ylabel("Original Prompt Symptom Tokens", fontsize=14)

layer_desc = f"layer {last_layer_index}" if last_layer_index is not None else "the last layer"
head_desc = f"all {num_attention_heads} heads" if num_attention_heads is not None else "all heads"
label_mode_desc = "romanized pinyin labels" if LABEL_RENDER_MODE == "pinyin" else "original Chinese labels"
caption = (
    "Attention source: average attention from each generated token to the selected prompt symptom tokens, "
    f"computed by averaging over {head_desc} in {layer_desc}; tick labels shown as {label_mode_desc}."
)
fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 1])
output_img = OUTPUT_DIR / "classic_token_heatmap.png"
plt.savefig(output_img, dpi=300)
print(f"✅ 经典文字热力图已保存至: {output_img}")
