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
