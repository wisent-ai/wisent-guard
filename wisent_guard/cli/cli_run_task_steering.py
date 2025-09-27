from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable

from wisent_guard.cli.cli_logger import setup_logger, bind
from wisent_guard.cli.cli_utils import (
    validate_or_explain,
)
from wisent_guard.cli.cli_activation import (
    make_collector,
    create_contrastive_pairs,
    extract_activations_for_pairs,
    build_pair_set_with_real_activations,
)

from wisent_guard.core import Model
from wisent_guard.cli.cli_steering import (
    build_steering,
    CAAConfig,
    HPRConfig,
    DACConfig,
    BiPOConfig,
    KSteeringConfig,
)
from wisent_guard.core.steering_methods.steering_evaluation import run_lm_harness_evaluation

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent_guard.cli.cli_data import LoadDataResult
    from wisent_guard.core.activations.activation_collection_method import ActivationCollectionLogic
    from wisent_guard.core import ContrastivePairSet
    from wisent_guard.cli.cli_performance import Trackers
    from wisent_guard.core.steering_methods.base import SteeringMethod



    SteeringConfig = CAAConfig | HPRConfig | DACConfig | BiPOConfig | KSteeringConfig


_LOG = setup_logger(__name__)


@dataclass(slots=True)
class ModelSettings:
    model_name: str
    device: str = "cuda"
    model_instance: Model | None = None


class DataSource(Enum):
    LM_EVAL = "lm_eval"
    CUSTOM = "custom"


@dataclass(slots=True)
class DataSettings:
    task_name: str = "hellaswag"
    split_ratio: float = 0.8
    limit: int = 10
    training_limit: int = 11
    testing_limit: int = 11
    seed: int = 42

@dataclass(slots=True)
class DataSettings:
    source: DataSource = DataSource.LM_EVAL
    task_names: tuple[str, ...] = ("hellaswag",)  
    custom_data_file: str | None = None        

    split_ratio: float = 0.8                       
    limit: int | None = None                       
    training_limit: int | None = None              
    testing_limit: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.source is DataSource.CUSTOM and not self.custom_data_file:
            raise ValueError("custom_data_file is required when source=CUSTOM.")
        if self.custom_data_file and self.source is DataSource.LM_EVAL:
            raise ValueError("custom_data_file provided but source is LM_EVAL; "
                             "set source=CUSTOM or remove the file path.")
        
@dataclass(slots=True)
class ActivationSettings:
    prompt_strategy: str = "multiple_choice"
    token_target: str = "choice_token"
    token_aggregation: str = "average"
    layer: int | str = 9


@dataclass(slots=True)
class SteeringSettings:
    steering_method: str = "CAA"
    steering_strength: float = 1.0

    config: SteeringConfig | None = None

    default_target_labels: tuple[int, ...] = (1,)
    default_avoid_labels: tuple[int, ...] = (0,)
    default_alpha: float = 0.5


@dataclass(slots=True)
class TrackingSettings:
    enable_memory_tracking: bool = False
    enable_latency_tracking: bool = False
    memory_sampling_interval: float = 1.0
    track_gpu_memory: bool = False
    show_memory_usage: bool = False
    show_timing_summary: bool = False
    verbose: bool = False


@dataclass(slots=True)
class EvaluationSettings:
    output_mode: str = "both"  


@dataclass(slots=True)
class PipelineSettings:
    model: ModelSettings
    data: DataSettings = field(default_factory=DataSettings)
    activation: ActivationSettings = field(default_factory=ActivationSettings)
    steering: SteeringSettings = field(default_factory=SteeringSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


def _validate(settings: PipelineSettings) -> dict[str, Any] | None:
    """Fast-fail validation; return error dict or None.
    """
    logger = bind(_LOG, stage="validate", task=settings.data.task_name)
    res = validate_or_explain(
        task_name=settings.data.task_name,
        verbose=settings.tracking.verbose,
    )
    if res.get("error"):
        logger.warning("validation_failed", extra={"error": res["error"]})
        return res
    logger.info("validation_ok")
    return None


def _build_or_reuse_model(ms: ModelSettings) -> Model:
    """Return a Model instance (reuse if already present).

    Arguments:
      - ms: ModelSettings with model_name/device/model_instance.
        Precisely:
            - model_name: Name of the model to load (if model_instance is None).
            - device: Device to put the model on (if model_instance is None).
            - model_instance: If not None, reuse this instance instead of loading a new one.

    Returns:
        - A Model instance, either reused or newly created.
    """
    logger = bind(_LOG, stage="model", model=ms.model_name)
    model = ms.model_instance if ms.model_instance is not None else Model(name=ms.model_name, device=ms.device)
    logger.info("model_ready", extra={"device": ms.device, "reuse": ms.model_instance is not None})
    return model


def _load_data(settings: PipelineSettings, model: Model | None) -> LoadDataResult:
    """
    Load data according to 'settings.data.source'.

    - LM_EVAL: uses 'load_train_test_lm_eval_format', requires 'model'
    - CUSTOM: uses 'load_train_test_custom_format', requires 'custom_data_file'

    arguments:
        settings: PipelineSettings with data section. Precisely:
            - data.source
            - data.task_names (if source=LM_EVAL)
            - data.custom_data_file (if source=CUSTOM)
            - data.split_ratio
            - data.limit
            - data.training_limit
            - data.testing_limit
            - data.seed
        model: Model instance, required if source=LM_EVAL.

    returns:
        A LoadDataResult with:
            - train_qa_pairs: ContrastivePairSet for training
            - test_qa_pairs_source: ContrastivePairSet for testing
            - task_type: Optional[str] with the task type (if any)
            - lm_task_data: Optional[dict[str, ConfigurableTask] | ConfigurableTask

        example:
         >>> from wisent_guard.cli_bricks.cli_run_task_steering import PipelineSettings, DataSettings, ModelSettings
         >>> from wisent_guard.core import Model
         >>> settings = PipelineSettings(
         ...     model=ModelSettings(model_name="llama3.1-8B-Instruct", device="cpu"),
         ...     data=DataSettings(
         ...         source=DataSource.LM_EVAL,
         ...         task_names=["hellaswag"],
         ...         split_ratio=0.8,
         ...         limit=100,
         ...         training_limit=80,
         ...         testing_limit=20,
         ...         seed=42,
         ...     ),
         ... )
            >>> model = Model(name=settings.model.model_name, device=settings.model.device)
            >>> data = _load_data(settings, model)
    """
   
    d = settings.data
    common_kwargs = dict(
        split_ratio=d.split_ratio,
        limit=d.limit,
        training_limit=d.training_limit,
        testing_limit=d.testing_limit,
        seed=d.seed,
    )

    logger = bind(_LOG, stage="data", task=d.task_name)
    logger.info("loading_data", extra={"source": d.source.value})

    data: LoadDataResult

    if d.source is DataSource.LM_EVAL:
        if model is None:
            raise ValueError("LM_EVAL source selected but 'model' is None.")
        try:
            from wisent_guard.cli.cli_data import load_train_test_lm_eval_format
        except Exception as e:
            raise ImportError("Failed to import lm-eval loader: load_train_test_lm_eval_format") from e

        data = load_train_test_lm_eval_format(
            model=model,
            task_names=list(d.task_names) if isinstance(d.task_names, Iterable) else [d.task_names],
            **common_kwargs,
        )

    elif d.source is DataSource.CUSTOM:
        try:
            from wisent_guard.cli.cli_data import load_train_test_custom_format
        except Exception as e:
            raise ImportError("Failed to import custom loader: load_train_test_custom_format") from e
        data = load_train_test_custom_format(
            file_path=str(d.custom_data_file),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown data source: {d.source}")

    train_qa_pairs: ContrastivePairSet = data.train_qa_pairs
    test_qa_pairs: ContrastivePairSet =  data.test_qa_pairs
    task_type: str | None = data.task_type
    lm_task_data: dict[str, ConfigurableTask] | ConfigurableTask | None = data.lm_task_data

    if train_qa_pairs is None or test_qa_pairs is None:
        raise ValueError("Loader did not return train_qa_pairs/test_qa_pairs.")

    logger.info("data_loaded", extra={"n_train": len(train_qa_pairs), "n_test": len(test_qa_pairs)})
    return LoadDataResult(
        train_qa_pairs=train_qa_pairs,
        test_qa_pairs=test_qa_pairs,
        task_type=task_type,
        lm_task_data=lm_task_data,
    )



def _compute_activations(
    settings: PipelineSettings,
    model: Model,
    qa_pairs: list[dict[str, Any]],
    trackers: Trackers | None,
) -> ContrastivePairSet:
    """Return a pair_set with real activations.

    Arguments:
      - settings: PipelineSettings with model/activation/tracking sections. 
      Precisely:
            - model.device
            - activation.prompt_strategy
            - activation.token_target
            - activation.layer
            - activation.token_aggregation
            - tracking.verbose
      - model: Model instance to use for activation collection.
      - qa_pairs: List of QA pairs to use for creating contrastive pairs and collecting activations.
      - trackers: Optional Trackers instance for memory/latency tracking (owned by caller).

    Returns:
        - A ContrastivePairSet with real activations attached.
    """
    logger = bind(_LOG, stage="activations", task=settings.data.task_name)
    collector: ActivationCollectionLogic = make_collector(model)
    contrastive_pairs = create_contrastive_pairs(
        collector=collector,
        prompt_construction_strategy=settings.activation.prompt_strategy,
        qa_pairs=qa_pairs,
        verbose=settings.tracking.verbose,
    )
    processed_pairs = extract_activations_for_pairs(
        collector=collector,
        contrastive_pairs=contrastive_pairs,
        layer=settings.activation.layer,
        device=settings.model.device,
        token_targeting_strategy=settings.activation.token_target,
        latency_tracker=trackers.latency if trackers else None, 
        verbose=settings.tracking.verbose,
    )
    pair_set = build_pair_set_with_real_activations(
        processed_pairs=processed_pairs,
        task_name=settings.data.task_name,
        verbose=settings.tracking.verbose,
    )
    logger.info("activations_ready", extra={"layer": settings.activation.layer})
    return pair_set


def _resolve_steering_config(
    settings: PipelineSettings,
) -> SteeringConfig:
    """Return a fully-populated SteeringConfig (patching device/paths as needed).

    Arguments:
      - settings: PipelineSettings with model/steering sections. 
      Precisely:
            - model.device
            - steering.config
            - steering.steering_method
            - steering.default_target_labels
            - steering.default_avoid_labels
            - steering.default_alpha

    Returns:
        - A SteeringConfig instance, fully populated.
    """
    st = settings.steering
    device = settings.model.device

    if st.config is not None:
        cfg = st.config
        patch: dict[str, Any] = {}
        if getattr(cfg, "device", None) is None:
            patch["device"] = device
        return replace(cfg, **patch) if patch else cfg

    method = st.steering_method
    if method == "CAA":
        return CAAConfig(method="CAA", device=device)
    if method == "HPR":
        return HPRConfig(method="HPR", device=device)
    if method == "DAC":
        return DACConfig(method="DAC", device=device)
    if method == "BiPO":
        return BiPOConfig(method="BiPO", device=device)
    # Default to KSteering
    return KSteeringConfig(
        method="KSteering",
        device=device,
        target_labels=st.default_target_labels,
        avoid_labels=st.default_avoid_labels,
        alpha=st.default_alpha,
    )


def _build_steering_obj(settings: PipelineSettings, pair_set: ContrastivePairSet) -> SteeringMethod:
    """Return a SteeringMethod instance

    Arguments:
      - settings: PipelineSettings with model/steering/tracking sections. 
      Precisely:
            - model.device
            - steering.steering_method
            - steering.steering_strength
            - tracking.verbose

    Returns:
        - An instance of the requested SteeringMethod.
    """
    logger = bind(_LOG, stage="steering", task=settings.data.task_name)
    cfg = _resolve_steering_config(settings)
    steering = build_steering(
        config=cfg,
        layer_idx=settings.activation.layer,
        pair_set=pair_set,
        verbose=settings.tracking.verbose,
    )
    logger.info("steering_ready", extra={"method": getattr(cfg, 'method', 'unknown')})
    return steering


def _evaluate(
    settings: PipelineSettings,
    model: Model,
    steering_obj: Any,
    task_data: dict[str, ConfigurableTask] | ConfigurableTask,
    test_qa_pairs_source: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return raw evaluation results.

    Arguments:
      - settings: PipelineSettings with model/steering/evaluation/tracking sections. 
      Precisely:
            - model.device
            - steering.steering_strength
            - evaluation.output_mode
            - tracking.verbose
      - model: Model instance to evaluate.
      - steering_obj: SteeringMethod instance to apply during evaluation.
      - task_data: Task object(s) to use for evaluation (from data loading).
      - test_qa_pairs_source: List of QA pairs for evaluation (from data loading).
    
    Returns:
        - Raw evaluation results as a dict which are:
            - accuracy: float accuracy over the test set
            - method: str, always "lm_harness_with_steering"
            - task_name: str, the name of the evaluated task
            - steering_applied: bool, whether steering was applied
            - full_results: dict, detailed results from the evaluation
            - baseline_likelihoods: list of floats, likelihoods without steering
            - steered_likelihoods: list of floats, likelihoods with steering

        Example of the output on truthfulqa_mc1:
            "result": {
                'accuracy': 0.24,
                'baseline_likelihoods': [-9.8203125, -9.4375, -10.078125, -15.1953125],
                'full_results': {'acc,none': 0.24,
                                 'acc_stderr,none': 'N/A',
                                 'alias': 'truthfulqa_mc1'},
                'method': 'lm_harness_with_steering',
                'steered_likelihoods': [-9.1796875, -8.359375, -8.953125, -14.4765625],
                'steering_applied': True,
                'task_name': 'truthfulqa_mc1'
            }
    """
    logger = bind(_LOG, stage="evaluation", task=settings.data.task_name)
    res: dict[str, Any] = run_lm_harness_evaluation(
        task_data=task_data,
        test_qa_pairs=test_qa_pairs_source,
        model=model,
        steering_methods=[steering_obj],
        layers=[settings.activation.layer],
        steering_strength=settings.steering.steering_strength,
        verbose=False,
        output_mode=settings.evaluation.output_mode,
    
    )
    logger.info("evaluation_done")
    return res


def run_task_steering_pipeline(
    settings: PipelineSettings,
    trackers: Trackers | None = None,
) -> dict[str, Any]:
    """
    Run the task-steering pipeline. The flow is:
    1) Validate settings (fast-fail) --> check task/model compatibility.
    2) Build or reuse the model instance.
    3) Load and prepare data (train/test split).
    4) Compute activations for the training data.
    5) Build the steering method and train it.
    6) Evaluate on the test set and return results.

    Arguments:
      - settings: PipelineSettings with all necessary configuration. 
      - trackers: Optional Trackers instance for memory/latency tracking (owned by caller).
    
    Returns:
        - Raw evaluation results as a dict (see _evaluate for details).
    """
    # 1) Validate (fast-fail)
    maybe_error = _validate(settings)
    if maybe_error:
        return maybe_error

    # 2) Model
    model = _build_or_reuse_model(settings.model)

    # 3) Data
    qa_pairs, test_qa_pairs_source, task_data = _load_data(settings, model)

    # 4) Activations
    pair_set = _compute_activations(settings, model, qa_pairs, trackers=trackers)

    # 5) Steering
    steering_obj = _build_steering_obj(settings, pair_set)

    # 6) Evaluate
    return _evaluate(
        settings,
        model=model,
        steering_obj=steering_obj,
        task_data=task_data,
        test_qa_pairs_source=test_qa_pairs_source,
    )

if __name__ == "__main__":
    model_path = "/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    settings = PipelineSettings(
        model=ModelSettings(model_name=model_path, device="cuda"),
        data=DataSettings(
            task_name="gsm8k",
            split_ratio=0.8,
            limit=10,
            training_limit=10,
            testing_limit=10,
            seed=42,
        ),
        activation=ActivationSettings(
            prompt_strategy="instruction_following",
            token_target="last_token",
            token_aggregation="average",
            layer=9,
        ),
        steering=SteeringSettings(
            steering_method="CAA",
            steering_strength=1.0,
            config=None,
            default_target_labels=(1,),
            default_avoid_labels=(0,),
            default_alpha=0.5,
        ),
    )

    results = run_task_steering_pipeline(settings, trackers=None)
    print("Results:", results)