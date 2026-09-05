import asyncio
import json
import stat

import pytest
from conftest import asgi_request

from market_gate.api import create_app
from market_gate.training_credentials import credential_status, read_credentials, save_credentials

FAKE = {"token_id": "ak-example-test-only", "token_secret": "as-example-secret-test-only"}


def test_save_preserves_other_settings_replaces_keys_and_is_private(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# keep this\nOTHER_SETTING=keep-me\nMODAL_TOKEN_ID=old\nMODAL_TOKEN_SECRET=old\n"
    )
    result = save_credentials(path, FAKE)
    assert result["configured"] is True
    assert set(result) == {"configured", "available"}
    assert "example" not in json.dumps(result)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    assert "# keep this\nOTHER_SETTING=keep-me\n" in text
    assert text.count("MODAL_TOKEN_ID=") == 1
    assert read_credentials(path)["MODAL_TOKEN_SECRET"] == FAKE["token_secret"]


@pytest.mark.parametrize("secret", ["line\ninjection", "", "a b", "x" * 257])
def test_invalid_credentials_never_echo_or_modify(tmp_path, secret):
    path = tmp_path / ".env"
    path.write_text("OTHER=preserve\n")
    with pytest.raises(ValueError) as error:
        save_credentials(path, FAKE | {"token_secret": secret})
    assert "as-example" not in str(error.value)
    assert path.read_text() == "OTHER=preserve\n"


def test_save_rejects_symlink(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    path = tmp_path / ".env"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        save_credentials(path, FAKE)
    assert target.read_text() == "preserve"


def test_api_saves_masked_status_and_blocks_hostile_origin(tmp_path):
    app = create_app()
    app.state.training_service.env_path = tmp_path / ".env"
    status, _ = asyncio.run(
        asgi_request(app, "POST", "/training/credentials", FAKE, origin="https://evil.example")
    )
    assert status == 403
    assert not (tmp_path / ".env").exists()
    status, body = asyncio.run(
        asgi_request(app, "POST", "/training/credentials", FAKE, origin="http://localhost:3010")
    )
    assert status == 200 and body["modal"]["configured"]
    status, settings = asyncio.run(asgi_request(app, "GET", "/training/settings"))
    assert status == 200
    assert FAKE["token_id"] not in json.dumps(settings)
    assert FAKE["token_secret"] not in json.dumps(settings)
    status, error = asyncio.run(
        asgi_request(
            app, "POST", "/training/credentials", FAKE | {"token_secret": "newline\nsecret"}
        )
    )
    assert status == 422 and "newline" not in json.dumps(error)


def test_missing_settings_exposes_only_status(tmp_path, monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert credential_status(tmp_path / ".env")["configured"] is False


@pytest.mark.parametrize(
    "payload", [{"learning_rate": 10**500}, {"backend": []}, {"hidden_1": 2048}]
)
def test_invalid_training_options_fail_before_launch(payload, monkeypatch):
    app = create_app()
    monkeypatch.setattr(app.state.training_service, "start", lambda *_: pytest.fail("launched"))
    status, body = asyncio.run(asgi_request(app, "POST", "/training/live/start", payload))
    assert status == 422
    assert body["detail"] == "Invalid model or training settings"
