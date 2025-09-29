__all__ = [
    "EXTRACTORS",
]
base_import: str = "wisent_guard.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."
EXTRACTORS: dict[str, str] = {
    # key → "module_path:ClassName" (supports dotted attr path after ':')
    "winogrande": f"{base_import}winogrande:WinograndeExtractor",
}
