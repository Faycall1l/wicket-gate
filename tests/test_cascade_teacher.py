from network_orch.cascade.actions import ActionKind as K
from network_orch.cascade.teacher import Teacher, build_prompt, parse_reply


def test_prompt_lists_only_narrowed_options_and_snippets():
    p = build_prompt("move it", [K.MIGRATE, K.SCALE], lambda t, k: [f"runbook for {k.value}"])
    assert "migrate, scale, refuse" in p and "monitor" not in p.split("Options:")[1].split("\n")[0]
    assert "[migrate] runbook for migrate" in p


def test_parse_accepts_wrapped_json_and_rejects_out_of_set():
    r = 'sure: {"kind":"scale","replicas":5,"to_zone":null,"confidence":0.8}'
    pr = parse_reply(r, [K.SCALE, K.MIGRATE])
    assert pr and pr.action.replicas == 5 and pr.confidence == 0.8
    assert parse_reply('{"kind":"monitor"}', [K.SCALE]) is None
    assert parse_reply("no json", [K.SCALE]) is None


def test_teacher_uses_injected_chat():
    t = Teacher(lambda s, u: '{"kind":"migrate","to_zone":"edge-b","confidence":0.7}')
    assert t.propose("move it", [K.MIGRATE]).action.to_zone == "edge-b"


def test_widens_once_when_reply_outside_narrowed_set():
    replies = iter(['{"kind":"monitor"}', '{"kind":"monitor"}'])
    t = Teacher(lambda s, u: next(replies))
    assert t.propose("is it ok", [K.SCALE]) .action.kind is K.MONITOR
    assert t.calls == 2


def test_parse_salvages_near_json():
    r = '{"kind":"migrate","replicas":null,"to_zone":"edge-a","namespace":null,"confidence":0.9"}'
    p = parse_reply(r, [K.MIGRATE])
    assert p and p.action.to_zone == "edge-a" and p.confidence == 0.9
