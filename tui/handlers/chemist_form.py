from typing import Mapping, Union, get_args, get_origin

from peptide_pipeline.chemist.agent_v1.config_chemist import ChemistConfig, RangeTarget


def is_optional(annotation) -> bool:
    origin = get_origin(annotation)
    return origin is Union and type(None) in get_args(annotation)


def is_range_target(annotation) -> bool:
    if annotation is RangeTarget:
        return True
    origin = get_origin(annotation)
    return origin is Union and RangeTarget in get_args(annotation)


def _base_type(annotation):
    if is_optional(annotation):
        return next(arg for arg in get_args(annotation) if arg is not type(None))
    return annotation


def _parse_scalar(raw_value: str, field_name: str, annotation):
    if raw_value is None or not raw_value.strip():
        if is_optional(annotation):
            return None
        raise ValueError(f"Field '{field_name}' is required")

    value = raw_value.strip()
    target_type = _base_type(annotation)

    try:
        if target_type is float:
            return float(value)
        if target_type is int:
            return int(value)
    except ValueError as exc:
        raise ValueError(f"Field '{field_name}' must be a valid {target_type.__name__}") from exc

    return value


def _parse_range_target(raw_inputs: Mapping[str, str], field_name: str):
    min_raw = (raw_inputs.get(f"{field_name}-min") or "").strip()
    max_raw = (raw_inputs.get(f"{field_name}-max") or "").strip()
    target_raw = (raw_inputs.get(f"{field_name}-target") or "").strip()
    values = [min_raw, max_raw, target_raw]

    if not any(values):
        return None

    if not min_raw or not max_raw:
        raise ValueError(f"Please provide min/max/target for {field_name} or leave all blank.")

    try:
        return RangeTarget(min=float(min_raw), max=float(max_raw), target=float(target_raw))
    except ValueError as exc:
        raise ValueError(f"min/max/target for '{field_name}' must be numeric.") from exc


def build_chemist_config_from_raw_inputs(raw_inputs: Mapping[str, str]) -> ChemistConfig:
    config_kwargs = {}

    for name, field in ChemistConfig.model_fields.items():
        if is_range_target(field.annotation):
            config_kwargs[name] = _parse_range_target(raw_inputs, name)
        else:
            config_kwargs[name] = _parse_scalar(raw_inputs.get(f"chem-{name}", ""), name, field.annotation)

    return ChemistConfig(**config_kwargs)
