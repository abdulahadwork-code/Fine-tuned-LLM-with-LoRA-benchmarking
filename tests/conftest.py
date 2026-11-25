import pytest
from domain_llm.data import Example

def make_examples(n_yes=30, n_no=20, n_maybe=10):
    out = []
    i = 0
    for label, n in (("yes", n_yes), ("no", n_no), ("maybe", n_maybe)):
        for _ in range(n):
            out.append(
                Example(
                    id=str(1000 + i),
                    question=f"Does treatment {i} work?",
                    contexts=(f"Background sentence {i}.", f"Results sentence {i}."),
                    label=label,
                    long_answer=f"Evidence suggests {label}.",
                )
            )
            i += 1
    return out


@pytest.fixture
def examples():
    return make_examples()
