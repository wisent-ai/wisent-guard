__all__ = [
    "EXTRACTORS",
]
base_import: str = "wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."
EXTRACTORS: dict[str, str] = {
    # key → "module_path:ClassName" (supports dotted attr path after ':')
    "winogrande": f"{base_import}winogrande:WinograndeExtractor",
    "piqa": f"{base_import}piqa:PIQAExtractor",
    "copa": f"{base_import}copa:COPAExtractor",
    "hellaswag": f"{base_import}hellaswag:HellaSwagExtractor",
    "swag": f"{base_import}swag:SWAGExtractor",
    "openbookqa": f"{base_import}openbookqa:OpenBookQAExtractor",
    "arc_easy": f"{base_import}arc_easy:Arc_EasyExtractor",
    "arc_challenge": f"{base_import}arc_challenge:Arc_ChallengeExtractor",
    "logiqa": f"{base_import}logiqa:LogiQAExtractor",
    "logiqa2": f"{base_import}logiqa2:LogiQA2Extractor",
    "wsc": f"{base_import}wsc:WSCExtractor",
    "mc-taco": f"{base_import}mc-taco:MCTACOExtractor",
    "social_iqa": f"{base_import}social_iqa:Social_IQAExtractor",
    "prost": f"{base_import}prost:PROSTExtractor",
    "pubmedqa": f"{base_import}pubmedqa:PubMedQAExtractor",
    "sciq": f"{base_import}sciq:SciQExtractor",
    "headqa_en": f"{base_import}headqa:HeadQAExtractor",
    "cb": f"{base_import}cb:CBExtractor",
    "mrpc": f"{base_import}mrpc:MRPCExtractor",
    "qnli": f"{base_import}qnli:QNLIExtractor",
    "qqp": f"{base_import}qqp:QQPExtractor",
    "rte": f"{base_import}rte:RTEExtractor",
    "sst2": f"{base_import}sst2:SST2Extractor",
    "wnli": f"{base_import}wnli:WNLIExtractor",
    "wic": f"{base_import}wic:WiCExtractor",
    "mutual": f"{base_import}mutual:MutualExtractor",
    }
