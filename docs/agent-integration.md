# Agent Integration

## Overview

Attach `codebase-indexer-py` to an agent session by building a skeleton string at startup, then registering three tools the agent can call on demand.

## Session setup

```python
from indexer.context import build_context
from indexer.tools import get_source, search, trace_callers

db_path = "~/.cache/codebase-indexer/my-app.db"

# Build the skeleton (auto-selects rendering mode by token budget)
skeleton = build_context(db_path, project="my-app", token_budget=8_000)

messages = [
    {
        "role": "system",
        "content": (
            "You are a software engineer working on this codebase.\n"
            "Use get_source(qn) to read function bodies, search(query) to find\n"
            "relevant code, and trace_callers(qn) to understand what depends on\n"
            "a given function before making changes.\n\n"
            f"Repository skeleton:\n\n{skeleton}"
        ),
    }
]

# Register the three tools with your agent framework and pass `messages`.
```

## Tools

### get_source

Returns the full source of a node, along with its direct callers and callees.

```python
result = get_source(db_path, "src.payments.service.charge")
```

### search

Full-text search across node names, signatures, and source using SQLite FTS5.

```python
results = search(db_path, "sql injection")
```

### trace_callers

Breadth-first search up the call graph to find everything that depends on a given node.

```python
callers = trace_callers(db_path, "src.payments.service.charge", depth=3)
```

## Skeleton example

At session start the agent receives a skeleton -- every file header, import list, and signature in the repo, grouped by file:

```txt
# my-app -- 42 files, 312 nodes  [skeleton]
# schema: Class=18 File=12 Function=156 Method=98 Interface=8 Type=20

### src/payments/service.py
# imports: stripe, src.payments.models, src.auth.models
def charge(user: User, amount_cents: int, currency: str) -> Payment:  # src.payments.service.charge
def refund(payment: Payment) -> bool:  # src.payments.service.refund

### src/payments/models.py
class Payment(BaseModel):  # src.payments.models.Payment
    def save(self) -> None:  # src.payments.models.Payment.save
    def to_dict(self) -> dict:  # src.payments.models.Payment.to_dict
```

Each line ends with a comment containing the fully-qualified name (FQN). The agent passes this FQN to `get_source()` to retrieve the full body.
