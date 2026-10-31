"""Static checks on isaac/g1_dexsuite.py (it cannot run without Isaac): the things that changed action or observation
sizes, or silently overrode settings, once already."""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "isaac" / "g1_dexsuite.py").read_text()
TREE = ast.parse(SRC)


def cls(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.ClassDef) and n.name == name)


def test_stored_command_legs_send_no_actions():
    """The policy must not see the legs as actions: dex-v6b has 9 arm + 1 pinch actions (2026-10-28: a missing
    action_dim = 0 made it 14 and the observation 20 larger, and every checkpoint load failed)."""
    c = cls("StoredCommandLowerBodyAction")
    props = [f for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "action_dim"]
    assert props, "StoredCommandLowerBodyAction.action_dim missing"
    ret = [n for n in ast.walk(props[0]) if isinstance(n, ast.Return)]
    assert ret and isinstance(ret[0].value, ast.Constant) and ret[0].value.value == 0
    assert any(isinstance(d, ast.Name) and d.id == "property" for d in props[0].decorator_list)


def test_no_action_cfg_field_shadows_isaaclab():
    """ActionTermCfg has clip (a dict), scale, asset_name...: our cfg fields must not reuse those names with other types."""
    c = cls("EMARelativeJointPositionActionCfg")
    fields = {t.target.id for t in c.body if isinstance(t, ast.AnnAssign)}
    assert "clip" not in fields and "rate_clip" in fields


def test_episode_length_set_after_default():
    """--walk_dr's 10 s episodes must come after the 6 s default in __post_init__ (else silently 6 s)."""
    post = next(f for f in cls("G1LiftEnvCfg").body if isinstance(f, ast.FunctionDef) and f.name == "__post_init__")
    src = ast.get_source_segment(SRC, post)
    assert src.index("self.episode_length_s = 10.0") > src.index("self.episode_length_s = 6.0")


def test_every_checkpoint_load_is_checked():
    """check_checkpoint() is called as many times as runner.load() (once before each load)."""
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)]
    loads = sum(isinstance(c.func, ast.Attribute) and c.func.attr == "load" and getattr(c.func.value, "id", "") == "runner" for c in calls)
    checks = sum(isinstance(c.func, ast.Name) and c.func.id == "check_checkpoint" for c in calls)
    assert loads >= 4 and checks == loads
