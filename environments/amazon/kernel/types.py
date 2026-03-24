"""Schema types for dependency-graph contracts and task intents (v2)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ToolClassification = Literal["causal", "knowledge-only", "stutter-only"]
RewardBasis = Literal["DB", "ENV_ASSERTION", "ACTION", "COMMUNICATE", "NL_ASSERTION"]


class ToolCallSpec(BaseModel):
    """How a tau2-style adapter executes this action: call a named tool with arguments."""

    tool_name: str
    arg_literals: dict[str, Any] = Field(default_factory=dict)
    arg_bindings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_call(self) -> "ToolCallSpec":
        if not self.tool_name.strip():
            raise ValueError("tool_call.tool_name cannot be empty")
        overlapping = set(self.arg_bindings) & set(self.arg_literals)
        if overlapping:
            raise ValueError(
                f"tool_call has overlapping keys in arg_bindings and "
                f"arg_literals: {sorted(overlapping)}"
            )
        for key in self.arg_literals:
            if not key.strip():
                raise ValueError("tool_call.arg_literals has empty key")
        for key, val in self.arg_bindings.items():
            if not key.strip():
                raise ValueError("tool_call.arg_bindings has empty key")
            if not val.strip():
                raise ValueError("tool_call.arg_bindings has empty binding id")
        return self


class BrowserActionSpec(BaseModel):
    """How a browser CUA adapter maps a kernel action to the web app.

    Every kernel action in a browser domain declares which page it occurs on
    and optionally which interactive DOM element it targets.

    - ``page``: the page type (route) where this action happens. Must match
      a ``page.type`` value in the domain's projection_fields. The Next.js
      app must have a route that renders this page.
    - ``element``: the ``data-action`` attribute value of the interactive
      DOM element the agent must interact with. If None, this is an
      observation or navigation action — no specific element, the agent
      just needs the page to render the right content.
    - ``interaction``: the primary browser primitive type for element
      actions (click, type, etc.). Required when ``element`` is set.
    - ``input_value``: dynamic input for type/select interactions,
      can reference entity fields via ``{entities.field_name}``.
    """

    page: str
    element: Optional[str] = None
    interaction: Optional[
        Literal[
            "click",
            "type",
            "keypress",
            "scroll",
            "select",
            "double_click",
        ]
    ] = None
    input_value: Optional[str] = None

    @model_validator(mode="after")
    def validate_browser_action(self) -> "BrowserActionSpec":
        if not self.page.strip():
            raise ValueError("browser_action.page cannot be empty")
        if self.element is not None and not self.element.strip():
            raise ValueError("browser_action.element cannot be blank when provided")
        if self.element is not None and self.interaction is None:
            raise ValueError(
                "browser_action.interaction is required when element is set"
            )
        if self.element is None and self.interaction is not None:
            raise ValueError(
                "browser_action.interaction should not be set without an element"
            )
        if self.input_value is not None and self.interaction not in (
            "type",
            "select",
            "keypress",
        ):
            raise ValueError(
                "browser_action.input_value is only valid for type/select/keypress interactions"
            )
        return self


def _check_unique(items: list[str], *, label: str) -> None:
    if len(items) != len(set(items)):
        raise ValueError(f"Duplicate values in {label}")


class ContextSlotSpec(BaseModel):
    """Named role slot to bind task instances to concrete entities."""

    slot_id: str
    entity_type: str
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_slot(self) -> "ContextSlotSpec":
        if not self.slot_id.strip():
            raise ValueError("context slot_id cannot be empty")
        if not self.entity_type.strip():
            raise ValueError("context entity_type cannot be empty")
        return self


class WorldPredicateSpec(BaseModel):
    """Predicate over projected world state."""

    op: Literal["eq", "neq", "gt", "lt", "gte", "lte"] = "eq"
    path: str
    value: Any

    @model_validator(mode="after")
    def validate_predicate(self) -> "WorldPredicateSpec":
        if not self.path.strip():
            raise ValueError("world predicate path cannot be empty")
        if self.op in {"gt", "lt", "gte", "lte"} and not isinstance(
            self.value, (int, float)
        ):
            raise ValueError(
                f"Comparison operator '{self.op}' requires numeric value, "
                f"got {type(self.value).__name__}"
            )
        return self


class WorldEffectSpec(BaseModel):
    """Forward world assignment effect."""

    path: str
    set: Any

    @model_validator(mode="after")
    def validate_effect(self) -> "WorldEffectSpec":
        if not self.path.strip():
            raise ValueError("world effect path cannot be empty")
        return self


class SyncEffectSpec(BaseModel):
    """Sync output: either copy from another path or set a literal."""

    path: str
    set: Optional[Any] = None
    from_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_sync_effect(self) -> "SyncEffectSpec":
        if not self.path.strip():
            raise ValueError("sync effect path cannot be empty")
        has_set = "set" in self.model_fields_set
        has_from_path = self.from_path is not None
        if not has_set and not has_from_path:
            raise ValueError("sync effect must have either 'set' or 'from_path'")
        if has_set and has_from_path:
            raise ValueError("sync effect cannot have both 'set' and 'from_path'")
        if self.from_path is not None and not self.from_path.strip():
            raise ValueError("sync effect from_path cannot be blank when provided")
        return self


class BindingSourceSpec(BaseModel):
    """How a binding value is extracted from a tool's return type."""

    binding_id: str
    source_tool: str
    extraction_path: str
    world_path: str
    observability_all_of: list[WorldPredicateSpec] = Field(default_factory=list)
    observability_any_of: list[WorldPredicateSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_binding_source(self) -> "BindingSourceSpec":
        if not self.binding_id.strip():
            raise ValueError("binding source binding_id cannot be empty")
        if not self.source_tool.strip():
            raise ValueError("binding source source_tool cannot be empty")
        if not self.extraction_path.strip():
            raise ValueError("binding source extraction_path cannot be empty")
        if not self.world_path.strip():
            raise ValueError("binding source world_path cannot be empty")
        return self


class BindingPredicateSpec(BaseModel):
    """Predicate over binding acquisition state."""

    binding_id: str
    acquired: bool = True

    @model_validator(mode="after")
    def validate_binding_predicate(self) -> "BindingPredicateSpec":
        if not self.binding_id.strip():
            raise ValueError("binding predicate binding_id cannot be empty")
        return self


class SyncRuleSpec(BaseModel):
    """Reactive rule that fires after init and after every action."""

    rule_id: str
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)
    effects_world: list[SyncEffectSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sync_rule(self) -> "SyncRuleSpec":
        if not self.rule_id.strip():
            raise ValueError("sync rule rule_id cannot be empty")
        if not self.effects_world:
            raise ValueError(
                f"Sync rule '{self.rule_id}' must have at least one effect"
            )
        return self


class ActionContract(BaseModel):
    """One transition/action contract in depgraph v2."""

    action_id: str
    requestor: Literal["assistant", "user"]
    classification: ToolClassification
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)
    requires_bindings: list[BindingPredicateSpec] = Field(default_factory=list)
    effects_world: list[WorldEffectSpec] = Field(default_factory=list)
    effects_bindings: list[str] = Field(default_factory=list)
    stutter_on_fail: bool = True

    # Adapter-specific runtime metadata (mutually exclusive per domain)
    tool_call: Optional[ToolCallSpec] = None
    browser_action: Optional[BrowserActionSpec] = None

    # Backward-compat shims: accept flat fields and migrate to tool_call
    tool_name: Optional[str] = Field(default=None, exclude=True)
    tool_arg_bindings: Optional[dict[str, str]] = Field(default=None, exclude=True)
    tool_arg_literals: Optional[dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_tool_fields(cls, data: Any) -> Any:
        """Auto-migrate legacy flat tool_name/tool_arg_* into tool_call block."""
        if not isinstance(data, dict):
            return data
        tool_name = data.pop("tool_name", None)
        arg_literals = data.pop("tool_arg_literals", None)
        arg_bindings = data.pop("tool_arg_bindings", None)
        if (
            tool_name is not None
            and data.get("tool_call") is None
            and data.get("browser_action") is None
        ):
            tc: dict[str, Any] = {"tool_name": tool_name}
            if arg_literals:
                tc["arg_literals"] = arg_literals
            if arg_bindings:
                tc["arg_bindings"] = arg_bindings
            data["tool_call"] = tc
        return data

    @model_validator(mode="after")
    def validate_semantics(self) -> "ActionContract":
        if not self.action_id.strip():
            raise ValueError("action_id cannot be empty")

        _check_unique(
            [predicate.binding_id for predicate in self.requires_bindings],
            label=f"action '{self.action_id}' requires_bindings",
        )
        _check_unique(
            self.effects_bindings,
            label=f"action '{self.action_id}' effects_bindings",
        )

        if self.classification == "stutter-only":
            if self.effects_world or self.effects_bindings:
                raise ValueError(
                    f"stutter-only action '{self.action_id}' cannot define world/binding effects"
                )

        # Validate tool_call binding references against requires_bindings
        if self.tool_call is not None:
            required_present_bindings = {
                predicate.binding_id
                for predicate in self.requires_bindings
                if predicate.acquired
            }
            for param_name, binding_id in self.tool_call.arg_bindings.items():
                if (
                    binding_id not in required_present_bindings
                    and self.classification != "knowledge-only"
                ):
                    raise ValueError(
                        f"Action '{self.action_id}' maps tool param '{param_name}' to binding "
                        f"'{binding_id}' that is not listed in requires_bindings"
                    )

        return self

    @property
    def resolved_tool_name(self) -> str | None:
        if self.tool_call is not None:
            return self.tool_call.tool_name
        return None

    @property
    def resolved_arg_bindings(self) -> dict[str, str]:
        if self.tool_call is not None:
            return self.tool_call.arg_bindings
        return {}

    @property
    def resolved_arg_literals(self) -> dict[str, Any]:
        if self.tool_call is not None:
            return self.tool_call.arg_literals
        return {}


class ActionSchemaVariantSpec(BaseModel):
    """One enum-like variant that expands to a concrete ActionContract."""

    variant_id: str
    action_id: Optional[str] = None
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)
    requires_bindings: list[BindingPredicateSpec] = Field(default_factory=list)
    effects_world: list[WorldEffectSpec] = Field(default_factory=list)
    effects_bindings: list[str] = Field(default_factory=list)
    stutter_on_fail: Optional[bool] = None

    # Adapter-specific overrides per variant
    tool_call: Optional[ToolCallSpec] = None
    browser_action: Optional[BrowserActionSpec] = None

    # Backward-compat shims
    tool_arg_bindings: Optional[dict[str, str]] = Field(default=None, exclude=True)
    tool_arg_literals: Optional[dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_tool_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        arg_literals = data.pop("tool_arg_literals", None)
        arg_bindings = data.pop("tool_arg_bindings", None)
        if (
            (arg_literals or arg_bindings)
            and data.get("tool_call") is None
            and data.get("browser_action") is None
        ):
            tc: dict[str, Any] = {"tool_name": "__variant__"}
            if arg_literals:
                tc["arg_literals"] = arg_literals
            if arg_bindings:
                tc["arg_bindings"] = arg_bindings
            data["tool_call"] = tc
        return data

    @model_validator(mode="after")
    def validate_variant(self) -> "ActionSchemaVariantSpec":
        if not self.variant_id.strip():
            raise ValueError("action schema variant_id cannot be empty")
        _check_unique(
            self.effects_bindings,
            label=f"action schema variant '{self.variant_id}' effects_bindings",
        )
        return self


class ActionSchemaSpec(BaseModel):
    """Authoring sugar for one conceptual tool with multiple concrete variants."""

    schema_id: str
    action_id_template: str
    variant_param: str = "variant_id"
    requestor: Literal["assistant", "user"]
    classification: ToolClassification
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)
    requires_bindings: list[BindingPredicateSpec] = Field(default_factory=list)
    effects_world: list[WorldEffectSpec] = Field(default_factory=list)
    effects_bindings: list[str] = Field(default_factory=list)
    stutter_on_fail: bool = True
    variants: list[ActionSchemaVariantSpec] = Field(default_factory=list)

    # Adapter-specific runtime metadata (schema level, merged with variant)
    tool_call: Optional[ToolCallSpec] = None
    browser_action: Optional[BrowserActionSpec] = None

    # Backward-compat shims
    tool_name: Optional[str] = Field(default=None, exclude=True)
    tool_arg_bindings: Optional[dict[str, str]] = Field(default=None, exclude=True)
    tool_arg_literals: Optional[dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_tool_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        tool_name = data.pop("tool_name", None)
        arg_literals = data.pop("tool_arg_literals", None)
        arg_bindings = data.pop("tool_arg_bindings", None)
        if (
            tool_name is not None
            and data.get("tool_call") is None
            and data.get("browser_action") is None
        ):
            tc: dict[str, Any] = {"tool_name": tool_name}
            if arg_literals:
                tc["arg_literals"] = arg_literals
            if arg_bindings:
                tc["arg_bindings"] = arg_bindings
            data["tool_call"] = tc
        return data

    @model_validator(mode="after")
    def validate_schema(self) -> "ActionSchemaSpec":
        if not self.schema_id.strip():
            raise ValueError("action schema_id cannot be empty")
        if not self.action_id_template.strip():
            raise ValueError(
                f"Action schema '{self.schema_id}' has empty action_id_template"
            )
        if not self.variant_param.strip():
            raise ValueError(
                f"Action schema '{self.schema_id}' has empty variant_param"
            )
        if not self.variants:
            raise ValueError(
                f"Action schema '{self.schema_id}' must declare at least one variant"
            )
        _check_unique(
            [variant.variant_id for variant in self.variants],
            label=f"action schema '{self.schema_id}' variants.variant_id",
        )
        return self

    def expand_actions(self) -> list[ActionContract]:
        """Expand schema variants into concrete ActionContract entries."""
        expanded: list[ActionContract] = []
        for variant in self.variants:
            substitutions = {
                "schema_id": self.schema_id,
                "variant_id": variant.variant_id,
                self.variant_param: variant.variant_id,
            }
            try:
                action_id = variant.action_id or self.action_id_template.format(
                    **substitutions
                )
            except KeyError as exc:
                raise ValueError(
                    f"Action schema '{self.schema_id}' action_id_template references unknown "
                    f"placeholder '{exc.args[0]}'"
                ) from exc

            merged_tool_call = self._merge_tool_call(variant)
            merged_browser_action = variant.browser_action or self.browser_action

            expanded.append(
                ActionContract(
                    action_id=action_id,
                    requestor=self.requestor,
                    classification=self.classification,
                    requires_world=[*self.requires_world, *variant.requires_world],
                    requires_bindings=[
                        *self.requires_bindings,
                        *variant.requires_bindings,
                    ],
                    effects_world=[*self.effects_world, *variant.effects_world],
                    effects_bindings=[
                        *self.effects_bindings,
                        *variant.effects_bindings,
                    ],
                    tool_call=merged_tool_call,
                    browser_action=merged_browser_action,
                    stutter_on_fail=(
                        self.stutter_on_fail
                        if variant.stutter_on_fail is None
                        else variant.stutter_on_fail
                    ),
                )
            )
        return expanded

    def _merge_tool_call(self, variant: ActionSchemaVariantSpec) -> ToolCallSpec | None:
        if self.tool_call is None and variant.tool_call is None:
            return None
        if self.tool_call is None:
            return variant.tool_call
        if variant.tool_call is None:
            return self.tool_call
        return ToolCallSpec(
            tool_name=self.tool_call.tool_name,
            arg_literals={
                **self.tool_call.arg_literals,
                **variant.tool_call.arg_literals,
            },
            arg_bindings={
                **self.tool_call.arg_bindings,
                **variant.tool_call.arg_bindings,
            },
        )


class TerminalProfileSpec(BaseModel):
    """Named terminal world profile used to constrain valid task end states."""

    profile_id: str
    description: Optional[str] = None
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def reject_binding_requirements(cls, data: Any) -> Any:
        if isinstance(data, dict) and "requires_bindings" in data:
            raise ValueError(
                "terminal profiles are world-only; remove 'requires_bindings' and "
                "model knowledge dependence through a final consuming action"
            )
        return data

    @model_validator(mode="after")
    def validate_profile(self) -> "TerminalProfileSpec":
        if not self.profile_id.strip():
            raise ValueError("terminal profile_id cannot be empty")
        if not self.requires_world:
            raise ValueError(
                f"terminal profile '{self.profile_id}' must declare at least one "
                "requires_world predicate"
            )
        return self


class TerminalSchemaVariantSpec(BaseModel):
    variant_id: str
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_variant(self) -> "TerminalSchemaVariantSpec":
        if not self.variant_id.strip():
            raise ValueError("terminal schema variant_id cannot be empty")
        if not self.requires_world:
            raise ValueError(
                f"terminal schema variant '{self.variant_id}' must declare "
                "at least one requires_world predicate"
            )
        return self


class TerminalSchemaDimensionSpec(BaseModel):
    dimension_id: str
    variants: list[TerminalSchemaVariantSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimension(self) -> "TerminalSchemaDimensionSpec":
        if not self.dimension_id.strip():
            raise ValueError("terminal schema dimension_id cannot be empty")
        if not self.variants:
            raise ValueError(
                f"terminal schema dimension '{self.dimension_id}' must declare "
                "at least one variant"
            )
        _check_unique(
            [v.variant_id for v in self.variants],
            label=f"terminal schema dimension '{self.dimension_id}' variants.variant_id",
        )
        return self


class TerminalSchemaSpec(BaseModel):
    schema_id: str
    profile_id_template: str
    description_template: str = ""
    requires_world: list[WorldPredicateSpec] = Field(default_factory=list)
    dimensions: list[TerminalSchemaDimensionSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schema(self) -> "TerminalSchemaSpec":
        if not self.schema_id.strip():
            raise ValueError("terminal schema_id cannot be empty")
        if not self.profile_id_template.strip():
            raise ValueError(
                f"Terminal schema '{self.schema_id}' has empty profile_id_template"
            )
        if not self.dimensions:
            raise ValueError(
                f"Terminal schema '{self.schema_id}' must declare at least one dimension"
            )
        _check_unique(
            [d.dimension_id for d in self.dimensions],
            label=f"terminal schema '{self.schema_id}' dimensions.dimension_id",
        )
        return self

    def expand_profiles(self) -> list[TerminalProfileSpec]:
        from itertools import product

        expanded: list[TerminalProfileSpec] = []
        for chosen_variants in product(*[d.variants for d in self.dimensions]):
            substitutions = {"schema_id": self.schema_id}
            substitutions.update(
                {
                    d.dimension_id: v.variant_id
                    for d, v in zip(self.dimensions, chosen_variants, strict=True)
                }
            )
            try:
                profile_id = self.profile_id_template.format(**substitutions)
            except KeyError as exc:
                raise ValueError(
                    f"Terminal schema '{self.schema_id}' profile_id_template references "
                    f"unknown placeholder '{exc.args[0]}'"
                ) from exc

            description = ""
            if self.description_template:
                try:
                    description = self.description_template.format(**substitutions)
                except KeyError:
                    description = self.description_template

            requires_world = [*self.requires_world]
            for variant in chosen_variants:
                requires_world.extend(variant.requires_world)

            expanded.append(
                TerminalProfileSpec(
                    profile_id=profile_id,
                    description=description or None,
                    requires_world=requires_world,
                )
            )
        return expanded


class GraphContractSpec(BaseModel):
    """Top-level depgraph v2 contract document."""

    version: int = 2
    context_slots: list[ContextSlotSpec] = Field(default_factory=list)
    projection_fields: list[str] = Field(default_factory=list)
    bindings: list[BindingSourceSpec] = Field(default_factory=list)
    actions: list[ActionContract] = Field(default_factory=list)
    action_schemas: list[ActionSchemaSpec] = Field(default_factory=list)
    sync_rules: list[SyncRuleSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def expand_action_schemas(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw_schemas = data.get("action_schemas") or []
        if not raw_schemas:
            return data

        expanded_actions: list[dict[str, Any]] = []
        for raw_schema in raw_schemas:
            schema = (
                raw_schema
                if isinstance(raw_schema, ActionSchemaSpec)
                else ActionSchemaSpec.model_validate(raw_schema)
            )
            expanded_actions.extend(
                action.model_dump(mode="python") for action in schema.expand_actions()
            )

        raw_actions = data.get("actions") or []
        normalized_actions: list[dict[str, Any] | ActionContract] = []
        for raw_action in raw_actions:
            if isinstance(raw_action, ActionContract):
                normalized_actions.append(raw_action.model_dump(mode="python"))
            else:
                normalized_actions.append(raw_action)

        payload = dict(data)
        payload["actions"] = [*normalized_actions, *expanded_actions]
        return payload

    @model_validator(mode="after")
    def validate_contract(self) -> "GraphContractSpec":
        if self.version != 2:
            raise ValueError(
                f"Graph contract version must be 2 for depgraph v2 (got {self.version})"
            )

        slot_ids = [s.slot_id for s in self.context_slots]
        _check_unique(slot_ids, label="context_slots.slot_id")

        projection_paths = [p for p in self.projection_fields if p.strip()]
        _check_unique(projection_paths, label="projection_fields")
        projection_set = set(projection_paths)

        binding_ids = [b.binding_id for b in self.bindings]
        _check_unique(binding_ids, label="bindings.binding_id")
        binding_id_set = set(binding_ids)
        for binding in self.bindings:
            if binding.world_path not in projection_set:
                raise ValueError(
                    f"Binding '{binding.binding_id}' world_path '{binding.world_path}' "
                    "must be listed in projection_fields"
                )

        action_ids = [a.action_id for a in self.actions]
        _check_unique(action_ids, label="actions.action_id")

        for action in self.actions:
            unknown_refs = (
                {predicate.binding_id for predicate in action.requires_bindings}
                | set(action.effects_bindings)
                | set(action.resolved_arg_bindings.values())
            ) - binding_id_set
            if unknown_refs:
                raise ValueError(
                    f"Action '{action.action_id}' references unknown binding ids: "
                    f"{sorted(unknown_refs)}"
                )

        binding_consumers: dict[str, set[str]] = {
            binding_id: set() for binding_id in binding_id_set
        }
        for action in self.actions:
            if action.classification == "knowledge-only":
                continue
            for binding_id in action.resolved_arg_bindings.values():
                binding_consumers.setdefault(binding_id, set()).add(action.action_id)
        for binding_id, consumer_actions in sorted(binding_consumers.items()):
            if consumer_actions:
                continue
            raise ValueError(
                f"Binding '{binding_id}' is never used as a downstream tool input "
                "via tool_call.arg_bindings"
            )

        sync_rule_ids = [rule.rule_id for rule in self.sync_rules]
        _check_unique(sync_rule_ids, label="sync_rules.rule_id")
        for rule in self.sync_rules:
            for predicate in rule.requires_world:
                if predicate.path not in projection_set:
                    raise ValueError(
                        f"Sync rule '{rule.rule_id}' requires_world references "
                        f"unknown projection path '{predicate.path}'"
                    )
            for effect in rule.effects_world:
                if effect.path not in projection_set:
                    raise ValueError(
                        f"Sync rule '{rule.rule_id}' effect writes to "
                        f"unknown projection path '{effect.path}'"
                    )
                if (
                    effect.from_path is not None
                    and effect.from_path not in projection_set
                ):
                    raise ValueError(
                        f"Sync rule '{rule.rule_id}' effect copies from "
                        f"unknown projection path '{effect.from_path}'"
                    )

        return self


class TaskIntent(BaseModel):
    """Task intent: a sampled (seed, terminal_profile) pair with BFS proof."""

    task_id: str
    start_world: list[WorldEffectSpec] = Field(default_factory=list)
    start_bindings: list[str] = Field(default_factory=list)
    goal_world: list[WorldPredicateSpec] = Field(default_factory=list)
    terminal_profile_id: Optional[str] = None
    required_actions: list[str] = Field(default_factory=list)
    min_plan_length: int = 1
    runtime: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def _strip_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("goal_bindings", None)
            data.pop("goal_capture_paths", None)
            data.pop("required_precedence", None)
        return data

    @model_validator(mode="after")
    def validate_task_intent(self) -> "TaskIntent":
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        _check_unique(
            self.start_bindings,
            label=f"task '{self.task_id}' start_bindings",
        )
        _check_unique(
            self.required_actions,
            label=f"task '{self.task_id}' required_actions",
        )
        if self.min_plan_length < 0:
            raise ValueError("min_plan_length must be >= 0")
        if (
            self.terminal_profile_id is not None
            and not self.terminal_profile_id.strip()
        ):
            raise ValueError("terminal_profile_id cannot be blank when provided")
        return self


class TaskSpecsDoc(BaseModel):
    """Top-level task specs document."""

    version: int = 1
    tasks: list[TaskIntent] = Field(default_factory=list)


class SamplingSeedSpec(BaseModel):
    """Seed state for task-intent sampling."""

    seed_id: str
    start_world: list[WorldEffectSpec] = Field(default_factory=list)
    start_bindings: list[str] = Field(default_factory=list)
    allowed_terminal_profiles: list[str] = Field(default_factory=list)
    min_depth: int = 3
    max_depth: int = 8

    @model_validator(mode="before")
    @classmethod
    def _strip_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("goal_capture_paths", None)
        return data

    @model_validator(mode="after")
    def validate_seed(self) -> "SamplingSeedSpec":
        if not self.seed_id.strip():
            raise ValueError("seed_id cannot be empty")
        if self.min_depth < 0:
            raise ValueError("min_depth must be >= 0")
        if self.max_depth < self.min_depth:
            raise ValueError("max_depth must be >= min_depth")
        _check_unique(
            self.start_bindings,
            label=f"seed '{self.seed_id}' start_bindings",
        )
        _check_unique(
            self.allowed_terminal_profiles,
            label=f"seed '{self.seed_id}' allowed_terminal_profiles",
        )
        return self


class SeedSchemaVariantSpec(BaseModel):
    variant_id: str
    start_world: list[WorldEffectSpec] = Field(default_factory=list)
    start_bindings: list[str] = Field(default_factory=list)
    min_depth_delta: int = 0
    max_depth_delta: int = 0

    @model_validator(mode="after")
    def validate_variant(self) -> "SeedSchemaVariantSpec":
        if not self.variant_id.strip():
            raise ValueError("seed schema variant_id cannot be empty")
        _check_unique(
            self.start_bindings,
            label=f"seed schema variant '{self.variant_id}' start_bindings",
        )
        return self


class SeedSchemaDimensionSpec(BaseModel):
    dimension_id: str
    variants: list[SeedSchemaVariantSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimension(self) -> "SeedSchemaDimensionSpec":
        if not self.dimension_id.strip():
            raise ValueError("seed schema dimension_id cannot be empty")
        if not self.variants:
            raise ValueError(
                f"seed schema dimension '{self.dimension_id}' must declare at least one variant"
            )
        _check_unique(
            [variant.variant_id for variant in self.variants],
            label=f"seed schema dimension '{self.dimension_id}' variants.variant_id",
        )
        return self


class SeedSchemaSpec(BaseModel):
    """Authoring sugar for expanding one family of concrete start seeds."""

    schema_id: str
    seed_id_template: str
    start_world: list[WorldEffectSpec] = Field(default_factory=list)
    start_bindings: list[str] = Field(default_factory=list)
    allowed_terminal_profiles: list[str] = Field(default_factory=list)
    min_depth: int = 3
    max_depth: int = 8
    dimensions: list[SeedSchemaDimensionSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _strip_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("goal_capture_paths", None)
        return data

    @model_validator(mode="after")
    def validate_schema(self) -> "SeedSchemaSpec":
        if not self.schema_id.strip():
            raise ValueError("seed schema_id cannot be empty")
        if not self.seed_id_template.strip():
            raise ValueError(
                f"Seed schema '{self.schema_id}' has empty seed_id_template"
            )
        if not self.allowed_terminal_profiles:
            raise ValueError(
                f"Seed schema '{self.schema_id}' must declare allowed_terminal_profiles"
            )
        if self.max_depth < self.min_depth:
            raise ValueError(
                f"Seed schema '{self.schema_id}' max_depth must be >= min_depth"
            )
        _check_unique(
            self.start_bindings,
            label=f"seed schema '{self.schema_id}' start_bindings",
        )
        _check_unique(
            self.allowed_terminal_profiles,
            label=f"seed schema '{self.schema_id}' allowed_terminal_profiles",
        )
        if not self.dimensions:
            raise ValueError(
                f"Seed schema '{self.schema_id}' must declare at least one dimension"
            )
        _check_unique(
            [dimension.dimension_id for dimension in self.dimensions],
            label=f"seed schema '{self.schema_id}' dimensions.dimension_id",
        )
        return self

    def expand_seeds(self) -> list[SamplingSeedSpec]:
        from itertools import product

        def _ordered_unique(values: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for value in values:
                if value in seen:
                    continue
                seen.add(value)
                out.append(value)
            return out

        expanded: list[SamplingSeedSpec] = []
        for chosen_variants in product(
            *[dimension.variants for dimension in self.dimensions]
        ):
            substitutions = {"schema_id": self.schema_id}
            substitutions.update(
                {
                    dimension.dimension_id: variant.variant_id
                    for dimension, variant in zip(
                        self.dimensions, chosen_variants, strict=True
                    )
                }
            )
            try:
                seed_id = self.seed_id_template.format(**substitutions)
            except KeyError as exc:
                raise ValueError(
                    f"Seed schema '{self.schema_id}' seed_id_template references unknown "
                    f"placeholder '{exc.args[0]}'"
                ) from exc

            start_world = [*self.start_world]
            start_bindings = list(self.start_bindings)
            min_depth = self.min_depth
            max_depth = self.max_depth
            for variant in chosen_variants:
                start_world.extend(variant.start_world)
                start_bindings.extend(variant.start_bindings)
                min_depth += variant.min_depth_delta
                max_depth += variant.max_depth_delta

            expanded.append(
                SamplingSeedSpec(
                    seed_id=seed_id,
                    start_world=start_world,
                    start_bindings=_ordered_unique(start_bindings),
                    allowed_terminal_profiles=list(self.allowed_terminal_profiles),
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
            )

        return expanded


class SamplingRequestDoc(BaseModel):
    """Input document for fan-out task-intent sampling."""

    version: int = 1
    max_tasks: int = 20
    terminal_profiles: list[TerminalProfileSpec] = Field(default_factory=list)
    terminal_schemas: list["TerminalSchemaSpec"] = Field(default_factory=list)
    seeds: list[SamplingSeedSpec] = Field(default_factory=list)
    seed_schemas: list[SeedSchemaSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _expand_schemas(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data.pop("goal_capture_paths", None)

        if data.get("seeds"):
            raise ValueError(
                "sampling_request direct 'seeds' authoring has been removed; use "
                "'seed_schemas' to generate concrete seeds"
            )

        raw_seed_schemas = data.get("seed_schemas") or []
        if raw_seed_schemas:
            expanded_seeds: list[dict[str, Any]] = []
            for raw_schema in raw_seed_schemas:
                schema = (
                    raw_schema
                    if isinstance(raw_schema, SeedSchemaSpec)
                    else SeedSchemaSpec.model_validate(raw_schema)
                )
                expanded_seeds.extend(
                    seed.model_dump(mode="python") for seed in schema.expand_seeds()
                )
            payload = dict(data)
            payload["seeds"] = expanded_seeds
        else:
            payload = dict(data)

        raw_terminal_schemas = payload.get("terminal_schemas") or []
        if raw_terminal_schemas:
            existing_profiles = list(payload.get("terminal_profiles") or [])
            for raw_schema in raw_terminal_schemas:
                schema = (
                    raw_schema
                    if isinstance(raw_schema, TerminalSchemaSpec)
                    else TerminalSchemaSpec.model_validate(raw_schema)
                )
                existing_profiles.extend(
                    profile.model_dump(mode="python")
                    for profile in schema.expand_profiles()
                )
            payload["terminal_profiles"] = existing_profiles

        return payload

    @model_validator(mode="after")
    def validate_sampling_request(self) -> "SamplingRequestDoc":
        if self.max_tasks <= 0:
            raise ValueError("max_tasks must be > 0")
        if not self.terminal_profiles:
            raise ValueError(
                "sampling_request must declare at least one terminal profile"
            )
        _check_unique(
            [profile.profile_id for profile in self.terminal_profiles],
            label="sampling_request.terminal_profiles.profile_id",
        )
        _check_unique(
            [seed.seed_id for seed in self.seeds],
            label="sampling_request.seeds.seed_id",
        )
        _check_unique(
            [schema.schema_id for schema in self.seed_schemas],
            label="sampling_request.seed_schemas.schema_id",
        )
        known_profiles = {profile.profile_id for profile in self.terminal_profiles}
        for seed in self.seeds:
            if not seed.allowed_terminal_profiles:
                raise ValueError(
                    f"seed '{seed.seed_id}' must declare allowed_terminal_profiles"
                )
            unknown_profiles = sorted(
                set(seed.allowed_terminal_profiles) - known_profiles
            )
            if unknown_profiles:
                raise ValueError(
                    f"seed '{seed.seed_id}' references unknown terminal profiles: {unknown_profiles}"
                )
        return self
