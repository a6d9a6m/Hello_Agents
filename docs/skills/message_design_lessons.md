## Message Design Lessons

1. Prefer framework-aligned models for core message objects.
Using `pydantic.BaseModel` makes validation, serialization, and future schema evolution explicit. It fits better than a hand-rolled dataclass when the object is a boundary type shared across agents, tools, logs, and external APIs.

2. Keep constructor ergonomics explicit before adding generic flexibility.
For `Message`, `content` and `role` are the true required fields, so they should stay as named constructor parameters. Extra optional data can still flow through `**kwargs`, but the primary call shape remains obvious and type-checker friendly.

3. Preserve rich internal fields while keeping transport output minimal.
`timestamp` and `metadata` are useful internally for tracing and future orchestration, but `to_dict()` should emit only the OpenAI-compatible wire format unless a caller explicitly needs richer serialization.

4. Put defaults inside the model boundary.
The model should decide how missing optional fields are filled, instead of requiring every call site to remember them. This keeps behavior consistent and reduces repetitive setup.

5. Use strict role constraints in both typing and runtime validation.
`Literal["user", "assistant", "system", "tool"]` captures the protocol at the type level, and Pydantic enforces it at runtime. This is better than accepting a broad `str` and relying on convention.

6. Avoid over-generalizing low-level primitives too early.
My earlier `**kwargs`-only dataclass constructor made the API more flexible than necessary and weakened readability. For foundational types, clarity beats genericity.
