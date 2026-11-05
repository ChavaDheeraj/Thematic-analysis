"""Any file with readable text is accepted, whatever its extension; audio, video and other non-text files are not."""
import io
import json
import zipfile

import pytest

from thematicml.data import LoadError, load_file
from thematicml.data.loaders import detect_format

PARAS = ["Travelling alone taught me to trust my own judgement in every new city.",
         "I felt unsafe at the station at night, so I took a registered taxi instead.",
         "A local family helped me find my hostel when my phone battery died."]


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return p


def texts(p, name=None):
    return [t for t in load_file(p, name).frame["text"]]


@pytest.mark.parametrize("name", ["notes.md", "diary.log", "chapter.rst", "interview", "field notes.TEXT", "script.py"])
def test_plain_text_with_any_extension_or_none(tmp_path, name):
    p = write(tmp_path, name, "\n\n".join(PARAS))
    assert detect_format(p) == "txt"
    assert texts(p) == PARAS


def test_utf16_text_files_are_read(tmp_path):
    for i, enc in enumerate(("utf-16", "utf-16-le")):  # Notepad's "Unicode", with and without a byte-order mark
        p = write(tmp_path, f"u{i}.txt", "\n\n".join(PARAS + ["Café, naïve — ½ price."]).encode(enc))
        assert texts(p)[-1] == "Café, naïve — ½ price."


def test_tsv_keeps_its_columns(tmp_path):
    p = write(tmp_path, "survey.tsv", "id\tanswer\n1\t" + PARAS[0] + "\n2\t" + PARAS[1] + "\n")
    lf = load_file(p)
    assert list(lf.frame.columns) == ["id", "answer"] and lf.frame["answer"].tolist() == PARAS[:2]


def test_subtitles_become_passages_without_timings(tmp_path):
    srt = """1
00:00:01,000 --> 00:00:03,500
Travelling alone taught me

2
00:00:03,600 --> 00:00:06,000
to trust my own judgement in every single new city that I visited over the past ten years, even when it was hard.

3
00:00:07,000 --> 00:00:09,000
<i>Then I went home.</i>
"""
    rows = load_file(write(tmp_path, "talk.srt", srt)).frame
    assert rows["text"].tolist() == ["Travelling alone taught me to trust my own judgement in every single new city that I visited over the past ten years, even when it was hard.",
                                     "Then I went home."]
    assert rows["cue_start"].tolist() == ["00:00:01,000", "00:00:07,000"]
    vtt = "WEBVTT\n\nNOTE recorded in 2024\nnot part of the talk\n\n00:01.000 --> 00:04.000\n<v Amber>Welcome to the show about travelling alone.\n"
    p = write(tmp_path, "talk.vtt", vtt)
    assert texts(p) == ["Welcome to the show about travelling alone."]
    assert detect_format(write(tmp_path, "captions", vtt)) == "vtt"  # recognised by content too


def test_rtf_text_without_control_words(tmp_path):
    rtf = (r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}{\colortbl;\red0\green0\blue0;}"
           r"{\*\generator Word;}\f0\fs24 Travelling alone taught me to trust myself.\par "
           r"Caf\'e9 owners were kind " + chr(92) + "u8212? always." + chr(92) + "par}")
    assert texts(write(tmp_path, "notes.rtf", rtf)) == ["Travelling alone taught me to trust myself.", "Café owners were kind — always."]


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_opendocument_text(tmp_path):
    content = ('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
               'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
               '<text:h>Interview one</text:h>' + "".join(f"<text:p>{t}</text:p>" for t in PARAS) +
               '<text:p/></office:text></office:body></office:document-content>')
    p = write(tmp_path, "interview.odt", zipped({"mimetype": "application/vnd.oasis.opendocument.text", "content.xml": content}))
    assert texts(p) == ["Interview one", *PARAS]


def test_powerpoint_one_record_per_slide(tmp_path):
    ns = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
    slide = lambda lines: f'<p:sld {ns}><p:cSld><p:spTree>' + "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in lines) + "</p:spTree></p:cSld></p:sld>"  # noqa: E731
    p = write(tmp_path, "findings.pptx", zipped({"ppt/slides/slide2.xml": slide([PARAS[2]]), "ppt/slides/slide10.xml": slide(["Last"]),
                                                  "ppt/slides/slide1.xml": slide(["Findings", PARAS[0]])}))
    f = load_file(p).frame
    assert f["text"].tolist() == [f"Findings\n{PARAS[0]}", PARAS[2], "Last"] and f["slide"].tolist() == [1, 2, 10]


def test_epub_chapters_in_reading_order(tmp_path):
    opf = ('<package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="c1" href="one.xhtml"/><item id="c2" href="two.xhtml"/></manifest>'
           '<spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>')
    container = '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>'
    page = lambda t: f"<html><body><h1>Chapter</h1><p>{t}</p><script>var x = 1;</script></body></html>"  # noqa: E731
    p = write(tmp_path, "memoir.epub", zipped({"META-INF/container.xml": container, "OEBPS/content.opf": opf,
                                               "OEBPS/one.xhtml": page(PARAS[0]), "OEBPS/two.xhtml": page(PARAS[1])}))
    f = load_file(p).frame
    assert f["text"].tolist() == ["Chapter", PARAS[1], "Chapter", PARAS[0]] and f["chapter"].tolist() == [1, 1, 2, 2]


def test_xml_elements_with_text(tmp_path):
    xml = ('<?xml version="1.0"?><responses><r id="1"><name>Ana</name><answer>' + PARAS[0] +
           '</answer></r><r id="2"><answer>I <em>loved</em> the night markets in Bangkok.</answer></r></responses>')
    assert texts(write(tmp_path, "export.xml", xml)) == ["Ana", PARAS[0], "I loved the night markets in Bangkok."]


def test_older_office_files_have_their_text_recovered(tmp_path):
    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(range(256)) * 4
    body = "\r".join(PARAS).encode("cp1252") + b"\x00\x01\x02" + "Default Paragraph Font".encode("cp1252") + bytes(512)
    lf = load_file(write(tmp_path, "old.doc", ole + bytes(64) + body + bytes(range(256))))
    assert lf.frame["text"].tolist() == PARAS and "best-effort" in lf.warnings[0]
    utf16 = ole + bytes(32) + PARAS[1].encode("utf-16-le") + bytes(32)  # text stored as UTF-16, as Word does for non-Latin text
    assert texts(write(tmp_path, "old16.doc", utf16)) == [PARAS[1]]


def test_a_mislabelled_file_is_read_by_its_content(tmp_path):
    html = "<html><body><table><tr><td>" + PARAS[0] + "</td></tr></table></body></html>"  # web apps often save tables as .xls
    p = write(tmp_path, "report.xls", html)
    assert detect_format(p) == "html" and texts(p) == [PARAS[0]]
    assert detect_format(write(tmp_path, "paper.txt", b"%PDF-1.4 rest")) == "pdf"


@pytest.mark.parametrize("name,data", [
    ("song.mp3", b"ID3\x03\x00\x00\x00rest"), ("interview.wav", b"RIFF\x00\x00\x00\x00WAVEfmt "), ("clip.mp4", b"\x00\x00\x00\x18ftypisom\x00\x00"),
    ("talk.webm", b"\x1aE\xdf\xa3rest"), ("voice.m4a", b"\x00\x00\x00\x20ftypM4A "), ("film.mov", b"text that is not really a film"),
    ("notes.txt", b"ID3\x03\x00\x00\x00\x00\x00\x01binary audio"), ("recording", b"OggS\x00\x02rest"),
])
def test_audio_and_video_are_refused(tmp_path, name, data):
    with pytest.raises(LoadError, match="Audio and video"):
        load_file(write(tmp_path, name, data))


@pytest.mark.parametrize("name,data,match", [
    ("photo.png", b"\x89PNG\r\n\x1a\nrest", "not a text file"), ("app.exe", b"MZ", "not a text file"),
    ("blob.xyz", bytes(range(256)) * 20, "No readable text"), ("photo", b"\xff\xd8\xff\xe0\x00\x10JFIF", "No readable text"),
    ("data.csv", bytes(range(256)) * 20, "readable text"),
])
def test_files_without_text_are_refused(tmp_path, name, data, match):
    with pytest.raises(LoadError, match=match):
        load_file(write(tmp_path, name, data))


def test_json_and_csv_still_tabular(tmp_path):
    p = write(tmp_path, "data.json", json.dumps([{"id": 1, "text": PARAS[0]}]))
    assert load_file(p).frame["text"].tolist() == [PARAS[0]]
