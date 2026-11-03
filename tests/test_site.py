"""The built website is served from one folder, and only from that folder."""
import importlib

import pytest


@pytest.fixture
def site(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("the site", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    monkeypatch.setenv("THEMATICML_FRONTEND", str(dist))
    monkeypatch.setenv("THEMATICML_HOME", str(tmp_path / "ws"))
    import thematicml_api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, tmp_path / "secret.txt"


def test_site_pages_and_files_are_served(site):
    client, _ = site
    assert client.get("/").text == "the site"
    assert client.get("/dashboard").text == "the site"  # routes belong to the single-page app
    assert client.get("/favicon.svg").text == "<svg/>"


@pytest.mark.parametrize("path", ["/..%2fsecret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt", "/assets/..%2f..%2fsecret.txt"])
def test_a_crafted_path_cannot_leave_the_build_folder(site, path):
    client, _ = site
    assert "private" not in client.get(path).text


def test_an_absolute_path_cannot_leave_the_build_folder(site):
    client, secret = site
    assert "private" not in client.get("/" + secret.as_posix()).text


@pytest.fixture
def locked(tmp_path, monkeypatch):
    """The app with a password, as it runs when other devices can reach it."""
    from fastapi.testclient import TestClient

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("the site", encoding="utf-8")
    monkeypatch.setenv("THEMATICML_FRONTEND", str(dist))
    monkeypatch.setenv("THEMATICML_HOME", str(tmp_path / "ws"))
    monkeypatch.setenv("THEMATICML_PASSWORD", "correct horse")
    monkeypatch.setenv("THEMATICML_REQUIRE_PASSWORD", "1")
    import thematicml_api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_without_the_password_nothing_opens(locked):
    for path in ("/", "/dashboard", "/api/projects", "/api/system"):
        r = locked.get(path)
        assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic"), path
    assert locked.post("/api/projects", json={"name": "Sneaky"}).status_code == 401


@pytest.mark.parametrize("auth", [("researcher", "wrong"), ("someone", "correct horse"), ("", "")])
def test_a_wrong_sign_in_is_refused(locked, auth):
    assert locked.get("/api/projects", auth=auth).status_code == 401


def test_a_malformed_sign_in_is_refused(locked):
    assert locked.get("/api/projects", headers={"Authorization": "Basic not-base64!"}).status_code == 401


def test_the_right_password_opens_pages_and_data(locked):
    auth = ("researcher", "correct horse")
    assert locked.get("/", auth=auth).text == "the site"
    made = locked.post("/api/projects", json={"name": "From a phone"}, auth=auth)
    assert made.status_code == 200
    assert [p["name"] for p in locked.get("/api/projects", auth=auth).json()] == ["From a phone"]


def test_the_health_check_stays_open_for_hosting(locked):
    assert locked.get("/api/health").json()["status"] == "ok"


def test_a_reachable_server_will_not_start_without_a_password(tmp_path, monkeypatch):
    monkeypatch.setenv("THEMATICML_HOME", str(tmp_path / "ws"))
    monkeypatch.delenv("THEMATICML_PASSWORD", raising=False)
    monkeypatch.setenv("THEMATICML_REQUIRE_PASSWORD", "1")
    import thematicml_api.main as main

    with pytest.raises(RuntimeError, match="THEMATICML_PASSWORD"):
        importlib.reload(main)
    monkeypatch.delenv("THEMATICML_REQUIRE_PASSWORD")
    importlib.reload(main)  # leave the module importable for later tests
