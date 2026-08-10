from PIL import Image, ImageDraw

from scripts.refine_dataset_v2_perceptual_groups import (
    build_anchor_groups,
    compute_phash,
    hash_distance,
    resolve_records,
)


def record(
    record_id,
    target="Potato Early blight",
    *,
    role="training_candidate",
    status="REVIEW",
    reason="PERCEPTUAL_LABEL_CONFLICT",
):
    return {
        "record_id": record_id,
        "dataset": "Synthetic",
        "role": role,
        "target_class": target,
        "mapping_status": "MATCHED",
        "width": "320",
        "height": "240",
        "cleaning_status": status,
        "exclusion_reason": reason,
    }


def signal(
    first,
    second,
    risk="SAME_TARGET",
    *,
    phash=2,
    good=100,
    ratio=0.5,
    inlier=0.9,
):
    return {
        "first_record_id": first,
        "second_record_id": second,
        "risk_type": risk,
        "dhash_distance": "1",
        "phash_distance": str(phash),
        "aspect_log_difference": "0.0",
        "first_keypoints": "200",
        "second_keypoints": "200",
        "orb_good_matches": str(good),
        "orb_match_ratio": str(ratio),
        "orb_inlier_ratio": str(inlier),
    }


def indexed(rows):
    return {row["record_id"]: row for row in rows}


def test_transitive_chaining_does_not_merge_non_anchor_match():
    groups = build_anchor_groups(
        ["rec_a", "rec_b", "rec_c"],
        [("rec_a", "rec_b"), ("rec_b", "rec_c")],
    )

    assert groups == {"rec_a": ["rec_a", "rec_b"]}


def test_representative_is_deterministic_lowest_direct_record_id():
    pairs = [("rec_c", "rec_b"), ("rec_b", "rec_a")]

    first = build_anchor_groups(["rec_c", "rec_b", "rec_a"], pairs)
    second = build_anchor_groups(["rec_a", "rec_b", "rec_c"], reversed(pairs))

    assert first == second == {"rec_a": ["rec_a", "rec_b"]}


def test_dct_phash_is_fixed_length_and_stable_for_resize():
    image = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 20, 70, 65), fill="black")
    draw.ellipse((75, 25, 115, 70), fill="green")

    original = compute_phash(image)
    resized = compute_phash(image.resize((512, 384), Image.Resampling.NEAREST))

    assert len(original) == 16
    assert len(resized) == 16
    assert hash_distance(original, resized) <= 2


def test_same_target_verified_near_duplicate_stays_include_and_is_grouped():
    records = {
        "rec_a": record("rec_a", status="INCLUDE", reason=""),
        "rec_b": record("rec_b", status="INCLUDE", reason=""),
    }
    rows, groups, human, _ = resolve_records(
        records,
        {"rec_a": "0" * 16, "rec_b": "0" * 15 + "1"},
        [signal("rec_a", "rec_b")],
    )
    result = indexed(rows)

    assert result["rec_a"]["resolved_cleaning_status"] == "INCLUDE"
    assert result["rec_b"]["resolved_cleaning_status"] == "INCLUDE"
    assert result["rec_a"]["refined_group_id"] == result["rec_b"]["refined_group_id"]
    assert result["rec_a"]["refined_similarity_status"] == "VERIFIED_NEAR_DUPLICATE"
    assert len(groups) == 1
    assert human == []


def test_weak_different_target_dhash_collision_is_restored_to_include():
    records = {
        "rec_a": record("rec_a"),
        "rec_b": record("rec_b", target="Potato healthy"),
    }

    rows, _, _, _ = resolve_records(
        records,
        {"rec_a": "0" * 16, "rec_b": "f" * 16},
        [],
    )

    assert {row["resolved_cleaning_status"] for row in rows} == {"INCLUDE"}
    assert {row["review_resolution"] for row in rows} == {
        "FALSE_POSITIVE_PERCEPTUAL_SCREEN"
    }


def test_strong_different_target_pair_excludes_both_records():
    records = {
        "rec_a": record("rec_a"),
        "rec_b": record("rec_b", target="Potato healthy"),
    }

    rows, groups, _, _ = resolve_records(
        records,
        {"rec_a": "0" * 16, "rec_b": "0" * 15 + "1"},
        [signal("rec_a", "rec_b", "DIFFERENT_TARGET")],
    )

    assert {row["resolved_cleaning_status"] for row in rows} == {"EXCLUDE"}
    assert {row["resolved_exclusion_reason"] for row in rows} == {
        "VERIFIED_PERCEPTUAL_LABEL_CONFLICT"
    }
    assert groups[0]["risk_status"] == "VERIFIED_LABEL_CONFLICT"


def test_benchmark_false_positive_restores_training_record():
    records = {
        "rec_train": record(
            "rec_train", reason="PERCEPTUAL_BENCHMARK_MATCH"
        ),
        "rec_test": record(
            "rec_test",
            role="locked_test",
            status="EXCLUDE",
            reason="LOCKED_BENCHMARK",
        ),
    }
    weak = signal(
        "rec_train",
        "rec_test",
        "TRAIN_TO_LOCKED_BENCHMARK",
        phash=16,
        good=2,
        ratio=0.01,
        inlier=0,
    )

    rows, _, _, _ = resolve_records(
        records,
        {"rec_train": "0" * 16, "rec_test": "f" * 16},
        [weak],
    )

    assert indexed(rows)["rec_train"]["resolved_cleaning_status"] == "INCLUDE"


def test_verified_benchmark_match_excludes_training_side_only():
    records = {
        "rec_train": record(
            "rec_train", reason="PERCEPTUAL_BENCHMARK_MATCH"
        ),
        "rec_test": record(
            "rec_test",
            role="locked_test",
            status="EXCLUDE",
            reason="LOCKED_BENCHMARK",
        ),
    }

    rows, groups, _, _ = resolve_records(
        records,
        {"rec_train": "0" * 16, "rec_test": "0" * 16},
        [signal("rec_train", "rec_test", "TRAIN_TO_LOCKED_BENCHMARK")],
    )
    result = indexed(rows)

    assert result["rec_train"]["resolved_exclusion_reason"] == (
        "VERIFIED_PERCEPTUAL_BENCHMARK_LEAKAGE"
    )
    assert result["rec_test"]["resolved_exclusion_reason"] == "LOCKED_BENCHMARK"
    assert groups[0]["risk_status"] == "VERIFIED_BENCHMARK_LEAKAGE"


def test_intermediate_signal_remains_small_human_review_case():
    records = {
        "rec_a": record("rec_a"),
        "rec_b": record("rec_b", target="Potato healthy"),
    }
    uncertain = signal(
        "rec_a",
        "rec_b",
        "DIFFERENT_TARGET",
        phash=14,
        good=20,
        ratio=0.06,
        inlier=0.55,
    )

    rows, _, human, _ = resolve_records(
        records,
        {"rec_a": "0" * 16, "rec_b": "0" * 15 + "1"},
        [uncertain],
    )

    assert {row["resolved_cleaning_status"] for row in rows} == {"REVIEW"}
    assert {row["review_resolution"] for row in rows} == {"STILL_UNCERTAIN"}
    assert len(human) == 1


def test_refinement_is_deterministic():
    records = {
        "rec_a": record("rec_a", status="INCLUDE", reason=""),
        "rec_b": record("rec_b", status="INCLUDE", reason=""),
    }
    phashes = {"rec_a": "0" * 16, "rec_b": "0" * 15 + "1"}
    signals = [signal("rec_a", "rec_b")]

    first = resolve_records(records, phashes, signals)
    second = resolve_records(records, phashes, list(reversed(signals)))

    assert first == second
