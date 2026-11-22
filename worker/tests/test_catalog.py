"""Catalog validation: good jobs pass, bad ones are refused with where=validate."""

import base64

import pytest

from fvstudio.catalog import FrameGrid
from fvstudio.errors import JobError

GOOD = {"engine": "h3", "model": "fasth3-v2", "task": "t2v", "prompt": "a fox", "width": 1344, "height": 768,
        "frames": 124, "seed": 7, "options": {"weights": "mxfp8", "pv8": True}, "upscale": 0, "client": {"job": "x"}}


def test_every_model_default_job_validates(catalog):
    for m in catalog.models:
        task = m.tasks[0]
        inputs = {}
        t = catalog.tasks[task]
        for k in t.required:
            inputs[k] = [{"url": "https://e/x.png"}] if k == "references" else {"url": "https://e/x.mp4"}
        j = catalog.validate({"engine": m.engine, "model": m.id, "task": task, "prompt": "p", "inputs": inputs})
        assert j.width, j.height == m.default_size
        assert j.frames == m.frames.default
        # every listed knob has an effective value (null defaults stay None)
        assert set(j.options) == set(m.knobs)


def test_good_job(catalog):
    j = catalog.validate(GOOD)
    assert j.model.id == "fasth3-v2" and j.seed == 7 and j.upscale == 0
    assert j.explicit == {"h3.weights": "mxfp8", "h3.pv8": True}
    assert j.options["h3.decoder_weights"] == "mxfp8"  # the knob default
    assert j.client == {"job": "x"}


def test_alias_resolves(catalog):
    j = catalog.validate({**GOOD, "model": "draft", "width": 832, "height": 480})
    assert j.model.id == "h3-plug4"


def test_full_option_ids_accepted(catalog):
    j = catalog.validate({**GOOD, "options": {"h3.weights": "native"}})
    assert j.options["h3.weights"] == "native"


@pytest.mark.parametrize(
    "patch,needle",
    [
        ({"frames": 125}, "17n+5"),
        ({"frames": 100}, "between"),
        ({"frames": 2000}, "between"),
        ({"width": 1000, "height": 768}, "sizes"),
        ({"options": {"nope": 1}}, "no option"),
        ({"options": {"vsa_sparsity": 0.3}}, "at least"),
        ({"options": {"dense_first_steps": 11}}, "at most"),
        ({"options": {"dense_first_steps": 1.5}}, "integer"),
        ({"options": {"weights": "fp4"}}, "one of"),
        ({"options": {"pv8": "maybe"}}, "true or false"),
        ({"task": "fl2v"}, "does not do task"),
        ({"engine": "nope"}, "engine"),
        ({"model": "nope"}, "unknown"),
        ({"prompt": ""}, "required"),
        ({"prompt": "x" * 2001}, "2000"),
        ({"negative_prompt": "ugly"}, "negative_prompt"),
        ({"upscale": 5}, "upscale"),
        ({"upscale": True}, "upscale"),
        ({"seed": -1}, "seed"),
        ({"op": "warm"}, "generate"),
    ],
)
def test_bad_jobs_refused(catalog, patch, needle):
    with pytest.raises(JobError) as e:
        catalog.validate({**GOOD, **patch})
    assert e.value.where == "validate"
    assert needle in str(e.value)


def test_an_older_pages_prompt_expansion_option_is_the_expand_prompt_knob(catalog):
    # The page deployed on 2026-11-18 sends fv-serve's `prompt_expansion` ("disabled" by
    # default) for every fastvideo model; the next day's image knows only `expand.prompt`.
    fv = lambda opts: {"engine": "fastvideo", "model": "ltx25-distill-dense", "prompt": "p", "options": opts}  # noqa: E731
    j = catalog.validate(fv({"prompt_expansion": "disabled"}))
    assert j.options["expand.prompt"] == "off" and j.explicit["expand.prompt"] == "off"
    j = catalog.validate(fv({"fv.prompt_expansion": "balanced"}))
    assert j.explicit["expand.prompt"] == "balanced"
    with pytest.raises(JobError, match="option prompt:"):
        catalog.validate(fv({"prompt_expansion": "sometimes"}))
    with pytest.raises(JobError, match="no option"):
        catalog.validate(fv({"nope": 1}))


def test_when_knob_with_default_is_dropped_not_refused(catalog):
    # The page sends every control; attention=auto leaves the preset's VSA, so vsa_sparsity stays.
    j = catalog.validate({**GOOD, "options": {"attention": "auto", "vsa_sparsity": 0.9, "dense_first_steps": 0}})
    assert j.explicit["h3.vsa_sparsity"] == 0.9
    assert "h3.dense_first_steps" in j.explicit  # at its default: harmless either way
    j = catalog.validate({**GOOD, "options": {"attention": "dense", "vsa_sparsity": 0.8}})
    assert "h3.vsa_sparsity" not in j.explicit
    with pytest.raises(JobError, match="applies only when"):
        catalog.validate({**GOOD, "options": {"attention": "dense", "vsa_sparsity": 0.9}})


def test_upscale_is_top_level_and_the_knob_is_ignored(catalog):
    j = catalog.validate({**GOOD, "upscale": 2, "options": {"upscale_factor": "4"}})
    assert j.upscale == 2 and "h3.upscale_factor" not in j.explicit
    with pytest.raises(JobError, match="upscales by 2, 4"):
        catalog.validate({"engine": "fastwan", "model": "fastwan-1.3b", "prompt": "p", "upscale": 3})
    with pytest.raises(JobError, match="no upscaler"):
        catalog.validate({"engine": "fastvideo", "model": "ltx25-distill-sol", "prompt": "p", "upscale": 2})


def test_media_rules(catalog):
    base = {"engine": "h3", "model": "h3-plug4", "task": "fl2v", "prompt": "p", "width": 832, "height": 480}
    with pytest.raises(JobError, match="needs input 'first_image'"):
        catalog.validate(base)
    j = catalog.validate({**base, "inputs": {"first_image": {"url": "https://e/a.png"}}})
    assert j.inputs["first_image"].kind == "url" and "last_image" not in j.inputs
    with pytest.raises(JobError, match="https"):
        catalog.validate({**base, "inputs": {"first_image": {"url": "http://e/a.png"}}})
    with pytest.raises(JobError, match="exactly one"):
        catalog.validate({**base, "inputs": {"first_image": {"url": "https://e/a.png", "r2": "uploads/x"}}})
    with pytest.raises(JobError, match="uploads/"):
        catalog.validate({**base, "inputs": {"first_image": {"r2": "jobs/x/video.mp4"}}})
    with pytest.raises(JobError, match="takes no input"):
        catalog.validate({**base, "inputs": {"first_image": {"url": "https://e/a.png"}, "audio": {"url": "https://e/a.wav"}}})
    small = base64.b64encode(b"x" * 100).decode()
    j = catalog.validate({**base, "inputs": {"first_image": {"b64": small, "mime": "image/png"}}})
    assert j.inputs["first_image"].inline_bytes == 100
    big = base64.b64encode(b"x" * (catalog.inline_limit + 1)).decode()
    with pytest.raises(JobError, match="MiB limit"):
        catalog.validate({**base, "inputs": {"first_image": {"b64": big, "mime": "image/png"}}})
    with pytest.raises(JobError, match="mime"):
        catalog.validate({**base, "inputs": {"first_image": {"b64": small}}})


def test_references_list(catalog):
    base = {"engine": "fastvideo", "model": "h3-ref2v-turbo", "task": "ref2v", "prompt": "p"}
    j = catalog.validate({**base, "inputs": {"references": [{"url": "https://e/a.png"}, {"r2": "uploads/b.mp4"}]}})
    assert [m.kind for m in j.inputs["references"]] == ["url", "r2"]
    with pytest.raises(JobError, match="needs input|non-empty"):
        catalog.validate({**base, "inputs": {"references": []}})
    with pytest.raises(JobError, match="at most 12"):
        catalog.validate({**base, "inputs": {"references": [{"url": "https://e/a.png"}] * 13}})


def test_fastvideo_null_default_knobs_absent_unless_set(catalog):
    j = catalog.validate({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "p"})
    assert j.options["fv.steps"] is None and j.explicit == {}
    j = catalog.validate({"engine": "fastvideo", "model": "wan22-ti2v-5b", "prompt": "p", "options": {"steps": 20}})
    assert j.explicit == {"fv.steps": 20}


def test_frame_grid():
    g = FrameGrid.parse({"grid": "12n+9", "min": 57, "max": 969, "default": 165, "choices": [165]})
    assert g.check(165) is None and g.check(57) is None
    assert "12n+9" in g.check(166)
    assert "between" in g.check(45)
