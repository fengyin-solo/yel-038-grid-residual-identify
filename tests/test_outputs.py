"""Acceptance tests for hidden-channel fingerprint reconciliation."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("GRID_RESIDUAL_ENTRY", "/app/grid_residual_identify.py")
PROFILES = [{'solver_id': 'GRID-A', 'fingerprint': [0.24, 0.81, 0.0074]},
 {'solver_id': 'GRID-B', 'fingerprint': [0.57, 1.43, 0.0121]},
 {'solver_id': 'GRID-C', 'fingerprint': [0.91, 0.62, 0.0048]},
 {'solver_id': 'GRID-D', 'fingerprint': [1.36, 1.08, 0.0186]},
 {'solver_id': 'GRID-E', 'fingerprint': [0.2726, 0.87965, 0.024881]},
 {'solver_id': 'GRID-F', 'fingerprint': [0.6781, 1.6499, 0.047673]},
 {'solver_id': 'GRID-G', 'fingerprint': [1.13845, 0.7919, 0.056736]},
 {'solver_id': 'GRID-H', 'fingerprint': [1.7816, 1.4288, 0.091436]},
 {'solver_id': 'GRID-I', 'fingerprint': [0.403, 1.15825, 0.094805]},
 {'solver_id': 'GRID-J', 'fingerprint': [0.8943, 2.0897, 0.118819]}]
READINGS = [{'slot': 'CH-3',
  'field_x': 0.24,
  'field_y': 2.025,
  'residual_norm': 0.00481,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-10',
  'field_x': 0.57,
  'field_y': 3.5749999999999997,
  'residual_norm': 0.007865,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-6',
  'field_x': 0.91,
  'field_y': 1.5499999999999998,
  'residual_norm': 0.00312,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-13',
  'field_x': 1.36,
  'field_y': 2.7,
  'residual_norm': 0.01209,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-9',
  'field_x': 0.2726,
  'field_y': 2.199125,
  'residual_norm': 0.01617265,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-5',
  'field_x': 0.6781,
  'field_y': 4.12475,
  'residual_norm': 0.03098745,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-12',
  'field_x': 1.13845,
  'field_y': 1.9797500000000001,
  'residual_norm': 0.036878400000000006,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-8',
  'field_x': 1.7816,
  'field_y': 3.572,
  'residual_norm': 0.059433400000000004,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-4',
  'field_x': 0.403,
  'field_y': 2.895625,
  'residual_norm': 0.061623250000000004,
  'step_size': 0.4,
  'temperature_c': 24.0},
 {'slot': 'CH-11',
  'field_x': 0.8943,
  'field_y': 5.22425,
  'residual_norm': 0.07723235,
  'step_size': 0.4,
  'temperature_c': 24.0}]
EXPECTED_ROWS = [{'slot': 'CH-10', 'solver_id': 'GRID-B', 'residual': 0.0},
 {'slot': 'CH-11', 'solver_id': 'GRID-J', 'residual': 0.0},
 {'slot': 'CH-12', 'solver_id': 'GRID-G', 'residual': 0.0},
 {'slot': 'CH-13', 'solver_id': 'GRID-D', 'residual': 0.0},
 {'slot': 'CH-3', 'solver_id': 'GRID-A', 'residual': 0.0},
 {'slot': 'CH-4', 'solver_id': 'GRID-I', 'residual': 0.0},
 {'slot': 'CH-5', 'solver_id': 'GRID-F', 'residual': 0.0},
 {'slot': 'CH-6', 'solver_id': 'GRID-C', 'residual': 0.0},
 {'slot': 'CH-8', 'solver_id': 'GRID-H', 'residual': 0.0},
 {'slot': 'CH-9', 'solver_id': 'GRID-E', 'residual': 0.0}]
OUTPUT_KEYS = ['mesh_id', 'status', 'solver_assignments', 'total_residual', 'accepted', 'diagnostic']
LEDGER_KEYS = ['mesh_id', 'input_sha256', 'result_sha256', 'total_residual']


def expose_candidate_path(path):
    if Path("/tmp") not in path.parents:
        return
    for ancestor in path.parents:
        if ancestor == Path("/tmp"):
            break
        ancestor.chmod(0o755)
    path.chmod(0o777)


def candidate_command(tmp_path, *arguments):
    expose_candidate_path(tmp_path)
    if os.geteuid() == 0 and Path("/usr/bin/setpriv").is_file():
        return [
            "/usr/bin/setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--",
            "/usr/local/bin/python3",
            ENTRY,
            *map(str, arguments),
        ]
    return [sys.executable, ENTRY, *map(str, arguments)]


def base_payload():
    return {
        "mesh_id": "GRID-STD-7",
        "limits": {"max_slots": 12, "residual_ceiling": 0.02},
        "profiles": copy.deepcopy(PROFILES),
        "readings": copy.deepcopy(READINGS),
    }


def invoke(tmp_path, payload=None, ledger=None, raw=None, output_name="result.json"):
    source = tmp_path / "readings.json"
    target = tmp_path / output_name
    if raw is None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    source.write_bytes(raw.encode("utf-8"))
    command = [str(value) for value in (source, target)]
    if ledger is not None:
        command.append(str(ledger))
    completed = subprocess.run(candidate_command(tmp_path, *command), capture_output=True, text=True, timeout=30)
    return completed, source, target


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_writes_aligned_output_quietly(tmp_path):
    """题面：「收敛报告成功写出时标准输出与标准错误均为空。」"""
    result, _, output = invoke(tmp_path, base_payload())
    assert result.returncode == 0
    assert result.stdout == "" and result.stderr == ""
    assert output.is_file()


def test_global_assignment_is_minimum_residual(tmp_path):
    """题面：「所有迭代槽位与求解器档案用全局最低总指纹差完成互配。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert [(row["slot"], row["solver_id"]) for row in document["solver_assignments"]] == [(row["slot"], row["solver_id"]) for row in EXPECTED_ROWS]
    assert [row["residual"] for row in document["solver_assignments"]] == pytest.approx([row["residual"] for row in EXPECTED_ROWS], abs=1e-12)


def test_assignment_rows_have_exact_fields(tmp_path):
    """题面：「solver_assignments 每行含 slot、solver_id、residual 三项。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert all(set(row) == {"slot", "solver_id", "residual"} for row in document["solver_assignments"])


def test_output_has_exact_top_level_keys(tmp_path):
    """题面：「报告根成员是 mesh_id、status、solver_assignments、total_residual、accepted、diagnostic。」"""
    result, _, output = invoke(tmp_path, base_payload())
    document = read(output)
    assert result.returncode == 0
    assert list(document) == OUTPUT_KEYS


def test_output_is_compact_utf8_with_single_lf(tmp_path):
    """题面：「结果序列化成紧凑 UTF-8 JSON，文件尾部恰有一个 LF。」"""
    result, _, output = invoke(tmp_path, base_payload())
    raw = output.read_bytes()
    assert result.returncode == 0
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b": " not in raw and b", " not in raw


def test_total_residual_matches_assignment(tmp_path):
    """题面：「total_residual 将所有行的 residual 相加得到。」以及公开三元指纹公式。"""
    payload = base_payload()
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    total = sum(row["residual"] for row in document["solver_assignments"])
    assert result.returncode == 0
    assert document["total_residual"] == pytest.approx(total, abs=1e-12)
    assert document["total_residual"] == pytest.approx(0.0, abs=1e-10)
    profiles = {
        row["solver_id"]: row["fingerprint"] for row in payload["profiles"]
    }
    readings = {row["slot"]: row for row in payload["readings"]}
    for assignment in document["solver_assignments"]:
        reading = readings[assignment["slot"]]
        temperature_factor = 1 + 0.0022 * (reading["temperature_c"] - 24)
        features = [
            reading["field_x"] / temperature_factor,
            reading["field_y"] * reading["step_size"],
            reading["residual_norm"] / (reading["step_size"] + 0.25),
        ]
        features = [round(value, 12) for value in features]
        expected = sum(
            ((left - right) / (abs(left) + abs(right) + 1e-12)) ** 2
            for left, right in zip(features, profiles[assignment["solver_id"]])
        )
        assert assignment["residual"] == pytest.approx(expected, abs=1e-12)


def test_ceiling_accepts_inside_case(tmp_path):
    """题面：「总差没有超过 residual_ceiling 时，accepted 是 true。」"""
    payload = base_payload()
    payload["limits"]["residual_ceiling"] = 1.0
    payload["profiles"][0]["fingerprint"] = [value * 1.04 for value in payload["profiles"][0]["fingerprint"]]
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0 and document["accepted"] is True
    assert document["status"] == "solvers_recovered"


def test_ceiling_quarantines_first_outside_value(tmp_path):
    """题面：「总差超过 residual_ceiling 时，结果继续写出并标记 quarantined。」"""
    payload = base_payload()
    payload["limits"]["residual_ceiling"] = 0.0001
    payload["profiles"][0]["fingerprint"] = [value * 1.04 for value in payload["profiles"][0]["fingerprint"]]
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0 and document["accepted"] is False
    assert document["status"] == "quarantined"


def test_missing_top_level_key_is_refused(tmp_path):
    """题面：「输入根少成员时以 1 退出，并写 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    payload.pop("limits")
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_extra_top_level_key_is_refused(tmp_path):
    """题面：「输入根多成员时以 1 退出，并写 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    payload["memo"] = "shift"
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_duplicate_top_level_key_is_refused(tmp_path):
    """题面：「同一对象键名出现两遍时，按坏输入输出 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = raw.replace('"mesh_id":"', '"mesh_id":"duplicate", "mesh_id":"', 1) + "\n"
    result, _, output = invoke(tmp_path, raw=raw)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_boolean_numeric_is_refused(tmp_path):
    """题面：「布尔值出现在数值位置时，输出 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    payload["readings"][0]["field_x"] = True
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_profile_fingerprint_length_is_refused(tmp_path):
    """题面：「fingerprint 不是三个正有限分量时，输出 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    payload["profiles"][0]["fingerprint"].pop()
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_duplicate_slot_is_refused(tmp_path):
    """题面：「两条迭代记录使用相同 slot 时，输出 数值拒收: BAD_SOLVER。」"""
    payload = base_payload()
    payload["readings"][1]["slot"] = payload["readings"][0]["slot"]
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()
    minimum = base_payload()
    minimum["mesh_id"] = "AB"
    minimum_result, _, minimum_output = invoke(tmp_path, minimum, output_name="minimum.json")
    assert minimum_result.returncode == 0 and read(minimum_output)["mesh_id"] == "AB"
    spaced = base_payload()
    spaced["mesh_id"] = "A B"
    spaced_result, _, spaced_output = invoke(tmp_path, spaced, output_name="spaced.json")
    assert spaced_result.returncode == 0 and read(spaced_output)["mesh_id"] == "A B"
    control = base_payload()
    control["mesh_id"] = "A\tB"
    control_result, _, control_output = invoke(tmp_path, control, output_name="control.json")
    assert control_result.returncode == 1 and control_result.stderr == "数值拒收: BAD_SOLVER\n" and not control_output.exists()


def test_three_slots_are_accepted(tmp_path):
    """题面：「程序仍需接受 3 至 12 个槽位的合法批次。」"""
    payload = base_payload()
    payload["limits"]["max_slots"] = 3
    payload["profiles"] = payload["profiles"][:3]
    payload["readings"] = payload["readings"][:3]
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0
    assert len(document["solver_assignments"]) == 3


def test_two_slots_are_refused(tmp_path):
    """题面：「少于 3 组或超过 max_slots 都属于坏输入。」"""
    payload = base_payload()
    payload["limits"]["max_slots"] = 3
    payload["profiles"] = payload["profiles"][:2]
    payload["readings"] = payload["readings"][:2]
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_twelve_slots_are_accepted(tmp_path):
    """题面：「三组与十二组都必须能形成完整对应」。"""
    payload = base_payload()
    payload["limits"]["max_slots"] = 12
    extra_profiles = [
        {"solver_id": "GRID-K", "fingerprint": [0.111, 0.222, 0.333]},
        {"solver_id": "GRID-L", "fingerprint": [1.111, 1.222, 1.333]},
    ]
    extra_readings = []
    for index, profile in enumerate(extra_profiles, start=1):
        fingerprint = profile["fingerprint"]
        extra_readings.append(
            {
                "slot": f"CH-X{index}",
                "field_x": fingerprint[0],
                "field_y": fingerprint[1] / 0.4,
                "residual_norm": fingerprint[2] * 0.65,
                "step_size": 0.4,
                "temperature_c": 24.0,
            }
        )
    payload["profiles"].extend(extra_profiles)
    payload["readings"].extend(extra_readings)
    result, _, output = invoke(tmp_path, payload)
    document = read(output)
    assert result.returncode == 0
    assert len(document["solver_assignments"]) == 12


def test_profile_count_above_max_slots_is_refused(tmp_path):
    """题面：「profiles 与 readings 数量必须相等、位于 3 到 max_slots 之间」。"""
    payload = base_payload()
    payload["limits"]["max_slots"] = 9
    result, _, output = invoke(tmp_path, payload)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not output.exists()


def test_ledger_has_exact_shape_and_is_idempotent(tmp_path):
    """题面：「完全相同的结果重新导入时，留痕簿字节不发生改变。」"""
    ledger = tmp_path / "trail.jsonl"
    result, _, output = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    first = ledger.read_bytes()
    second, _, _ = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    item = json.loads(first)
    assert result.returncode == 0 and second.returncode == 0
    assert list(item) == LEDGER_KEYS and first.endswith(b"\n")
    assert ledger.read_bytes() == first and second.stdout == "" and second.stderr == ""


def test_different_result_appends_one_ledger_line(tmp_path):
    """题面：「有效结果发生变化时，留痕簿在末尾按成功次序多一行。」"""
    ledger = tmp_path / "trail.jsonl"
    first, _, _ = invoke(tmp_path, payload=base_payload(), ledger=ledger)
    changed = base_payload()
    changed["mesh_id"] = "GRID-STD-8"
    second, _, _ = invoke(tmp_path, payload=changed, ledger=ledger)
    assert first.returncode == 0 and second.returncode == 0
    assert len(ledger.read_bytes().splitlines()) == 2
    assert ledger.read_bytes().endswith(b"\n")


def test_ledger_path_equal_to_output_is_refused(tmp_path):
    """题面：「留痕目标与结果目标重名时，输出 残差簿拒收: BAD_GRID_LEDGER。」"""
    payload = base_payload()
    source = tmp_path / "readings.json"
    target = tmp_path / "same.json"
    source.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    result = subprocess.run(candidate_command(tmp_path, source, target, target), capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "残差簿拒收: BAD_GRID_LEDGER\n" and not target.exists()


def test_io_failure_preserves_existing_target(tmp_path):
    """题面：「结果无法写入时，终端写 归档失败: GRID_IO，旧结果保持可读。」"""
    target = tmp_path / "occupied"
    target.mkdir()
    source = tmp_path / "readings.json"
    source.write_text(json.dumps(base_payload(), separators=(",", ":")), encoding="utf-8")
    result = subprocess.run(candidate_command(tmp_path, source, target), capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "归档失败: GRID_IO\n"
    assert target.exists() and target.is_dir() and not any(target.iterdir()) and "traceback" not in result.stderr.lower()


def test_wrong_argument_count_is_refused(tmp_path):
    """题面：「命令行参数数量超出 3 或 4 时，退出 1 并写 数值拒收: BAD_SOLVER。」"""
    target = tmp_path / "result.json"
    result = subprocess.run(candidate_command(tmp_path), capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and result.stderr == "数值拒收: BAD_SOLVER\n" and not target.exists()
