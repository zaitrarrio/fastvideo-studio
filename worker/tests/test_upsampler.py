"""Prompt expansion: the llama-server argv reproduces fv-upsampler.sh's
flags, the family prompts, cleaning and the length caps, the vision part,
the fallbacks (500 / timeout / connection refused), the VRAM estimate and
the catalog knob."""

from __future__ import annotations

import base64
import socket

import pytest

from fvstudio import upsampler as up
from fvstudio.sidecar import Sidecar
from fvstudio.upsampler import (
    ALIAS, CTX, H3_DESCRIPTION, H3_MUSIC, H3_SOUNDSCAPE, KNOB, MAX_CHARS, MODEL_FILE, MMPROJ_FILE, TREE, Upsampler,
    clean, clean_for, clean_h3, cut, family_of, kv_bytes, llama_argv, llama_probe, mode_of, plan_sidecar, system_prompt, vram_gib,
)
from tests.fake_llama import FakeLlama

# ---- the sidecar's command line ------------------------------------------------------


def test_argv_reproduces_the_launcher_flags():
    """fv-upsampler.sh `launch`, with the Studio's context and one slot."""
    argv = llama_argv("/opt/fastvideo-rs/bin/llama-server", "/w/m.gguf", "/w/mmproj.gguf", 18080)
    assert argv[:3] == ["/opt/fastvideo-rs/bin/llama-server", "-m", "/w/m.gguf"]
    assert argv[3:5] == ["--mmproj", "/w/mmproj.gguf"]
    rest = " ".join(argv[5:])
    assert rest == ("--host 127.0.0.1 --port 18080 --alias qwen3-vl-4b-instruct -ngl 99 -c 16384 -np 1 --no-kv-unified "
                    "-fa on -ctk q8_0 -ctv q8_0 --cache-reuse 256 --cache-ram 4096 --fit off --no-webui")
    # Text only: no --mmproj.
    assert "--mmproj" not in llama_argv("b", "m", None, 1)
    # The launcher's own settings reproduce too.
    a = llama_argv("b", "m", "p", 18080, ctx=65536, parallel=2)
    assert a[a.index("-c") + 1] == "65536" and a[a.index("-np") + 1] == "2"


def test_plan_sidecar_is_pure_and_names_the_tree_files():
    argv, env = plan_sidecar("/opt/fastvideo-rs/bin/llama-server", "/workspace/w/qwen3-vl-4b-instruct-gguf", 18080, env={"FV_WEIGHTS": "/workspace/w"})
    assert argv[2] == f"/workspace/w/qwen3-vl-4b-instruct-gguf/{MODEL_FILE}"
    assert argv[4] == f"/workspace/w/qwen3-vl-4b-instruct-gguf/{MMPROJ_FILE}"
    assert env == {"FV_WEIGHTS": "/workspace/w"}
    argv, env = plan_sidecar("b", "/t/", 1, vision=False)
    assert "--mmproj" not in argv and env == {}
    assert TREE == "r2:fv-weights/weights/qwen3-vl-4b-instruct-gguf=qwen3-vl-4b-instruct-gguf"
    # A Sidecar can be built from the plan (not started).
    sc = Sidecar(up.ENGINE, up.sidecar_key("/t"), argv, env, llama_probe("http://127.0.0.1:1"))
    assert sc.engine == "upsampler" and not sc.alive()


# ---- memory --------------------------------------------------------------------------


def test_vram_estimate():
    # The launcher's note: ~4.8 GB of q8_0 KV at 65536 tokens.
    assert 4.7 <= kv_bytes(65536) / 2**30 <= 4.9
    assert 1.15 <= kv_bytes(16384) / 2**30 <= 1.25
    # Weights 3.99 GiB + projector 0.78 + KV 1.20 + compute 0.5.
    assert vram_gib() == pytest.approx(6.5, abs=0.05)
    assert up.VRAM_GIB == vram_gib(CTX)
    assert vram_gib(vision=False) < vram_gib()
    assert vram_gib(ctx=65536) >= vram_gib(ctx=16384) + 3.5
    assert 4.7 <= up.FETCH_GIB <= 4.8


# ---- families and system prompts ----------------------------------------------------------


def test_family_selection():
    assert family_of("h3") == "h3" and family_of("fv-h3") == "h3"
    assert family_of("fastwan") == "wan" and family_of("wan") == "wan"
    assert family_of("ltx") == "ltx"
    assert family_of("something") == "other" and family_of(None) == "other"


def test_system_prompts_follow_the_family():
    """rewrite.rs `system_prompts_follow_the_family`, plus the budgets."""
    wan = system_prompt("wan", "balanced")
    assert "makes no sound" in wan and "60 to 100 words" in wan and "1800 characters" in wan
    assert "One English paragraph" in wan
    ltx = system_prompt("ltx", "quality", image=True)
    assert "Start directly with the main action" in ltx and "140 to 200 words" in ltx and "image you cannot see" in ltx
    assert "3600 characters" in ltx
    h3 = system_prompt("h3", "balanced")
    assert "camera language" in h3 and "<d>[English]" in h3 and "image you cannot see" not in h3
    assert H3_DESCRIPTION in h3 and H3_SOUNDSCAPE in h3 and H3_MUSIC in h3
    assert "one paragraph" in h3 and "[Shot 1]" not in h3 and "60 to 110 words" in h3 and "3200 characters" in h3
    assert "the three labels exactly as shown" in h3 and "One English paragraph" not in h3
    q = system_prompt("h3", "quality")
    assert "[Shot 1]" in q and "120 to 220 words" in q
    other = system_prompt("other", "balanced")
    assert "detailed video prompt" in other and "80 to 130 words" in other
    # A seen image (vision) against an unseen one.
    seen = system_prompt("wan", "balanced", image=True, vision=True)
    assert "attached image" in seen and "cannot see" not in seen
    # The negative prompt is never rewritten; the LLM is told what it excludes.
    neg = system_prompt("wan", "balanced", negative="blurry,  low   quality")
    assert "The user excludes: blurry, low quality." in neg
    assert "excludes" not in wan


# ---- cleaning ---------------------------------------------------------------------------


def test_clean_strips_wrappers():
    assert clean("  A cat walks.  ", 100) == "A cat walks."
    assert clean('"A cat walks."', 100) == "A cat walks."
    assert clean("Prompt: A cat\nwalks.", 100) == "A cat walks."
    assert clean("Rewritten Prompt: A cat walks.", 100) == "A cat walks."
    assert clean("```text\nA cat walks.\n```", 100) == "A cat walks."
    assert clean("```\nA cat walks.\n```", 100) == "A cat walks."
    assert clean("<think>hmm</think>\n**A cat**  walks.", 100) == "**A cat** walks."
    # Inner quotes keep the wrapper (it is dialogue, not a wrapper).
    assert clean('"Go," she says, "now."', 100) == '"Go," she says, "now."'
    assert clean("   ", 100) is None
    assert clean('""', 100) is None
    # Whitespace collapses into one paragraph.
    assert clean("A fox\n\n  runs   fast.\n", 100) == "A fox runs fast."


def test_cut_prefers_sentences_then_words():
    s = "One two three. Four five six. Seven eight nine."
    assert cut(s, 100) == s
    assert cut(s, 32) == "One two three. Four five six."
    assert cut("Alpha beta gamma delta. Epsilon", 12) == "Alpha beta"
    assert cut("Supercalifragilistic", 5) == "Super"
    assert len(cut("word " * 1000, 1800)) <= 1800


def test_family_caps():
    long = "A fox runs. " * 500  # 6000 chars
    for fam, limit in MAX_CHARS.items():
        out = clean_for(fam, long)
        assert out and len(out) <= limit, fam
    assert MAX_CHARS == {"wan": 1800, "ltx": 3600, "h3": 3200, "other": 1800}
    assert len(clean_for("wan", long)) <= 1800 and clean_for("wan", long).endswith(".")


def test_h3_answers_keep_their_sections():
    a = "```\nintegrated_multimodal_description: A keeper climbs.\n  Mara says <d>[English] Light it.</d>\noverall_soundscape: Wind.\nnon_diegetic_music: None.\n```"
    assert clean_h3(a, 10_000) == ("integrated_multimodal_description: A keeper climbs.\nMara says <d>[English] Light it.</d>\n"
                                   "overall_soundscape: Wind.\nnon_diegetic_music: None.")
    assert clean_h3("A cat walks.", 10_000) == "integrated_multimodal_description: A cat walks."
    q = "integrated_multimodal_description:\n[Shot 1] A wide shot.\n[Shot 2] A close-up.\noverall_soundscape: Rain.\nnon_diegetic_music: Strings."
    assert clean_h3(q, 10_000) == ("integrated_multimodal_description: [Shot 1] A wide shot.\n[Shot 2] A close-up.\n"
                                   "overall_soundscape: Rain.\nnon_diegetic_music: Strings.")
    assert clean_h3("  ", 10_000) is None


def test_h3_answers_fit_the_limit():
    """Over the limit: the music goes to None, then the description is cut,
    never inside <d>…</d>, the soundscape kept."""
    a = ("integrated_multimodal_description: A keeper climbs the stairs. Mara says <d>[English] Light the lamp now please.</d> "
         "The beam sweeps the sea.\noverall_soundscape: Wind and rain.\nnon_diegetic_music: A low cello drone.")
    full = clean_h3(a, 10_000)
    n = len(full)
    p = clean_h3(a, n - 1)
    assert p.endswith("non_diegetic_music: None.") and "The beam sweeps the sea." in p, p
    # Down to a limit that still holds the labels, the soundscape and a few
    # words of description (~120 characters).
    for limit in range(n - 2, 120, -1):
        p = clean_h3(a, limit)
        assert len(p) <= limit, p
        assert p.count("<d>") == p.count("</d>"), p
        assert "overall_soundscape: Wind and rain." in p, p
        assert p.startswith("integrated_multimodal_description: ") and p.endswith("non_diegetic_music: None."), p


# ---- the client: request shape, vision, fallbacks ------------------------------------------------

PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64).decode()


def test_expand_sends_the_family_prompt_and_cleans_the_answer():
    with FakeLlama(answer='"A red fox trots across fresh snow at dawn."\n') as f:
        u = Upsampler(f.base)
        out = u.expand("a fox", "fastwan", "balanced")
        assert out == "A red fox trots across fresh snow at dawn."
        assert u.last_error is None and u.calls == 1 and u.failures == 0
        req = f.last
        assert req["model"] == ALIAS and req["temperature"] == 0.6 and req["max_tokens"] == 512
        assert "makes no sound" in f.system_of() and "60 to 100 words" in f.system_of()
        assert f.user_of() == "a fox"
        # Quality mode, H3 family through the catalog's fv-h3 id.
        u.expand("a fox", "fv-h3", "quality")
        assert "[Shot 1]" in f.system_of() and f.last["max_tokens"] == 1024
        # Off / empty: no call.
        assert u.expand("a fox", "h3", "off") == "a fox" and u.expand("  ", "h3", "quality") == "  "
        assert len(f.requests) == 2


def test_h3_answer_keeps_sections_through_expand():
    ans = "integrated_multimodal_description: A fox runs.\noverall_soundscape: Wind.\nnon_diegetic_music: None."
    with FakeLlama(answer=ans) as f:
        out = Upsampler(f.base).expand("a fox", "h3", "balanced", negative=None)
        assert out == ans


def test_vision_part_when_an_image_is_given_and_the_projector_is_loaded():
    with FakeLlama() as f:
        u = Upsampler(f.base, vision=True)
        u.expand("make it run", "ltx", "balanced", image_b64=PNG_B64, image_mime="image/png")
        user = f.user_of()
        assert isinstance(user, list) and user[0] == {"type": "text", "text": "make it run"}
        assert user[1]["type"] == "image_url" and user[1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"
        assert "attached image" in f.system_of() and "cannot see" not in f.system_of()
        # Text-only server: the image stays out, the system prompt says it is unseen.
        Upsampler(f.base, vision=False).expand("make it run", "ltx", "balanced", image_b64=PNG_B64)
        assert f.user_of() == "make it run" and "image you cannot see" in f.system_of()
        # No image: neither line.
        Upsampler(f.base, vision=True).expand("make it run", "ltx", "balanced")
        assert "first frame" not in f.system_of()
        # An image over the data-URL cap is not attached.
        Upsampler(f.base, vision=True).expand("x", "ltx", "balanced", image_b64="A" * (up.MAX_IMAGE_DATA_URL + 1))
        assert f.user_of() == "x" and "image you cannot see" in f.system_of()


def test_negative_prompt_reaches_the_system_prompt_only():
    with FakeLlama() as f:
        Upsampler(f.base).expand("a fox", "wan", "balanced", negative="blurry")
        assert "The user excludes: blurry." in f.system_of() and f.user_of() == "a fox"


def test_fallback_on_500():
    with FakeLlama(fail_status=500) as f:
        logs = []
        u = Upsampler(f.base, log=logs.append)
        assert u.expand("a fox", "wan", "balanced") == "a fox"
        assert u.last_error and u.last_error.startswith("HTTP 500") and u.failures == 1
        assert logs and "prompt expansion failed" in logs[0] and "original prompt" in logs[0]


def test_fallback_on_timeout():
    with FakeLlama(hang_s=1.5) as f:
        u = Upsampler(f.base)
        assert u.expand("a fox", "h3", "quality", timeout_s=0.3) == "a fox"
        assert u.last_error and "timeout" in u.last_error.lower() or "timed out" in (u.last_error or "")


def test_fallback_on_connection_refused():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    u = Upsampler(f"http://127.0.0.1:{port}")
    assert u.expand("a fox", "ltx", "balanced") == "a fox"
    assert u.last_error and ("refused" in u.last_error.lower() or "connect" in u.last_error.lower())


def test_fallback_on_empty_or_garbage_answers():
    with FakeLlama(answer="   ") as f:
        u = Upsampler(f.base)
        assert u.expand("a fox", "wan", "balanced") == "a fox"
        assert "nothing left" in (u.last_error or ""), u.last_error
    with FakeLlama(answer='""') as f:
        assert Upsampler(f.base).expand("a fox", "wan", "balanced") == "a fox"


# ---- the readiness probe -----------------------------------------------------------------


def test_llama_probe_health_then_one_token():
    with FakeLlama(loading=True) as f:
        probe = llama_probe(f.base)
        assert probe() == (False, None) and f.requests == []
        f.loading = False
        assert probe() == (True, None)
        assert f.requests[-1]["max_tokens"] == 1 and f.requests[-1]["model"] == ALIAS
    assert llama_probe("http://127.0.0.1:1")() == (False, None)


# ---- the catalog knob ------------------------------------------------------------------------


def test_every_model_lists_the_knob_and_the_worker_reads_it(catalog):
    knob = catalog.knobs[KNOB]
    assert knob.type == "select" and knob.default == "off" and knob.options == ("off", "balanced", "quality")
    assert knob.group == "quality" and knob.cli is None and knob.fv_field is None
    assert "Qwen3-VL-4B" in catalog.data["knobs"][KNOB]["help"] and "llama-server" in catalog.data["knobs"][KNOB]["help"]
    for m in catalog.models:
        assert KNOB in m.knobs, m.id
    # Short name (what the page sends) and full id both validate; default off.
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "options": {"prompt": "quality"}})
    assert j.options[KNOB] == "quality" and KNOB in j.explicit and mode_of(j) == "quality"
    j = catalog.validate({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "a fox", "options": {"expand.prompt": "balanced"}})
    assert mode_of(j) == "balanced"
    j = catalog.validate({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox"})
    assert mode_of(j) == "off" and KNOB not in j.explicit
    with pytest.raises(Exception, match="must be one of"):
        catalog.validate({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "a fox", "options": {"prompt": "max"}})
    # The knob never reaches the engines' flags / fields.
    from fvstudio import flags
    from fvstudio.flags import EngineSettings
    s = EngineSettings(h3_bin="h3", fastwan_bin="fastwan", backend="cpu")
    j = catalog.validate({"engine": "h3", "model": "fasth3-v2", "prompt": "a fox", "options": {"prompt": "quality"}})
    assert not any("expand" in a or a == "quality" for a in flags.h3_serve_argv(j, s, catalog.knobs, 1))
    j = catalog.validate({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "a fox", "options": {"prompt": "quality"}})
    body = flags.fastvideo_body(j, catalog.knobs, {}, 1)
    assert "prompt_expansion" not in body and "expand" not in body and body["prompt"] == "a fox"
