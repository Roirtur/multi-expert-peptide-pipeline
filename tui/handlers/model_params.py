from textual.widgets import Checkbox, Input, Label, Select


def get_model_fields(model_configs, model_name: str, include_hyperparameters: bool = False):
    cfg = model_configs.get(model_name, {})
    fields = list(cfg.get("params", []))
    if include_hyperparameters:
        fields.extend(cfg.get("hyperparameters", []))
    return fields


async def mount_model_fields(container, fields, id_prefix: str = ""):
    for child in list(container.children):
        await child.remove()

    for field in fields:
        field_type = field.get("type", "input")
        widget_id = f"{id_prefix}{field['id']}"
        if field_type == "input":
            await container.mount(
                Label(field["label"]),
                Input(placeholder=field.get("placeholder", ""), id=widget_id),
            )
        elif field_type == "select":
            await container.mount(
                Label(field["label"]),
                Select([(option, option) for option in field.get("options", [])], id=widget_id),
            )
        elif field_type == "checkbox":
            await container.mount(Checkbox(field["label"], id=widget_id))


def _coerce_param_value(value, spec):
    coerce = spec.get("coerce")
    if coerce == "int":
        return int(value)
    if coerce == "float":
        return float(value)
    return value


def collect_model_params(root, fields, query_in_root, id_prefix: str = ""):
    params = {}
    for field in fields:
        widget_id = f"#{id_prefix}{field['id']}"
        field_type = field.get("type", "input")
        if field_type == "input":
            widget = query_in_root(root, widget_id, Input)
            value = widget.value or field.get("placeholder", "")
        elif field_type == "select":
            widget = query_in_root(root, widget_id, Select)
            value = widget.value
        elif field_type == "checkbox":
            widget = query_in_root(root, widget_id, Checkbox)
            value = widget.value
        else:
            continue
        params[field["id"]] = _coerce_param_value(value, field)
    return params
