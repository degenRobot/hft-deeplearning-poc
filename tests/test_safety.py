from pathlib import Path


def test_no_execution_route_literals() -> None:
    source = "\n".join(path.read_text() for path in Path("src").rglob("*.py"))
    forbidden = ("/api/v3/order", "@userData", "listenKey")
    assert not any(item in source for item in forbidden)
