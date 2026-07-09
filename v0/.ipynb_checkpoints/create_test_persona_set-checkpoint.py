import argparse
import csv
import random
import re
from datetime import datetime
from pathlib import Path


PROFILE_COLUMNS = ["user_id", "name", "username", "user_char", "description"]
DEFAULT_COUNT = 4


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create a small IO/normal persona test set and preserve a mapping "
            "from simulation user IDs to the original tweet dataset user IDs."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment folder containing personas_io_drivers.csv and personas_normal_users.csv.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Name for a new timestamped output experiment folder when --output-dir is omitted.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=base_dir / "experiments",
        help="Base experiments folder used for timestamped output folders.",
    )
    parser.add_argument(
        "--source-experiment-dir",
        type=Path,
        default=None,
        help="Source experiment folder containing the full persona CSVs. Defaults to --experiment-dir.",
    )
    parser.add_argument("--io-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--normal-count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Randomly sample rows instead of taking the first rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to <experiment-dir>/test_4_each.",
    )
    parser.add_argument(
        "--action-probabilities-source",
        type=Path,
        default=None,
        help="Optional probability CSV keyed by original user_id/source_author_id.",
    )
    parser.add_argument(
        "--action-probabilities-output",
        type=Path,
        default=None,
        help="Optional output probability CSV remapped to simulation user IDs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=base_dir,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def normalize_experiment_name(experiment_name: str | None) -> str:
    if not experiment_name:
        experiment_name = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("Experiment name must include at least one letter or number.")
    return normalized


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_source_author_ids(
    profile_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    audit_by_username = {row.get("username", ""): row for row in audit_rows}
    audit_by_user_id = {row.get("user_id", ""): row for row in audit_rows}

    merged_rows = []
    for row in profile_rows:
        merged = dict(row)
        audit_row = (
            audit_by_username.get(row.get("username", ""))
            or audit_by_user_id.get(row.get("user_id", ""))
            or {}
        )
        merged["source_author_id"] = (
            row.get("source_author_id")
            or audit_row.get("source_author_id")
            or ""
        )
        merged_rows.append(merged)
    return merged_rows


def select_rows(
    rows: list[dict[str, str]],
    count: int,
    rng: random.Random,
    sample: bool,
) -> list[dict[str, str]]:
    if count <= 0:
        raise ValueError("Counts must be greater than 0.")
    if count > len(rows):
        raise ValueError(f"Requested {count} rows, but only {len(rows)} are available.")
    if sample:
        return rng.sample(rows, count)
    return rows[:count]


def renumber_profile_rows(
    rows: list[dict[str, str]],
    start_user_id: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    profile_rows = []
    mapping_rows = []
    for offset, row in enumerate(rows):
        simulation_user_id = str(start_user_id + offset)
        profile_row = dict(row)
        original_persona_user_id = profile_row.get("user_id", "")
        profile_row["user_id"] = simulation_user_id
        profile_rows.append(profile_row)
        mapping_rows.append(
            {
                "simulation_user_id": simulation_user_id,
                "persona_user_id": original_persona_user_id,
                "source_author_id": profile_row.get("source_author_id", ""),
                "username": profile_row.get("username", ""),
                "name": profile_row.get("name", ""),
            }
        )
    return profile_rows, mapping_rows


def remap_action_probabilities(
    source_path: Path,
    output_path: Path,
    mapping_rows: list[dict[str, str]],
) -> None:
    probability_rows = read_csv(source_path)
    probabilities_by_original_id = {
        row.get("user_id", ""): row
        for row in probability_rows
    }
    output_rows = []
    missing_ids = []

    for mapping in mapping_rows:
        source_author_id = mapping["source_author_id"]
        probability_row = probabilities_by_original_id.get(source_author_id)
        if probability_row is None:
            missing_ids.append(source_author_id)
            continue
        output_row = dict(probability_row)
        output_row["user_id"] = mapping["simulation_user_id"]
        output_row["source_author_id"] = source_author_id
        output_rows.append(output_row)

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise KeyError(
            f"Missing probability rows for {len(missing_ids)} source_author_id values: {preview}"
        )

    fieldnames = ["user_id", "source_author_id", "p_action", "post", "reply", "retweet"]
    write_csv(output_path, output_rows, fieldnames)


def load_persona_rows(experiment_dir: Path, label: str) -> list[dict[str, str]]:
    profile_path = experiment_dir / f"personas_{label}.csv"
    audit_path = experiment_dir / f"personas_{label}_audit.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"Persona CSV not found: {profile_path}")
    profile_rows = read_csv(profile_path)
    if audit_path.exists():
        return merge_source_author_ids(profile_rows, read_csv(audit_path))
    return profile_rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_experiment_dir = args.source_experiment_dir or args.experiment_dir
    if source_experiment_dir is None:
        raise ValueError(
            "Provide --source-experiment-dir or --experiment-dir with the full persona CSVs."
        )
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.experiment_dir is not None:
        output_dir = args.experiment_dir / "test_4_each"
    else:
        output_dir = args.experiments_dir / normalize_experiment_name(args.experiment_name)

    io_rows = select_rows(
        load_persona_rows(source_experiment_dir, "io_drivers"),
        args.io_count,
        rng,
        args.sample,
    )
    normal_rows = select_rows(
        load_persona_rows(source_experiment_dir, "normal_users"),
        args.normal_count,
        rng,
        args.sample,
    )

    io_profile_rows, io_mapping_rows = renumber_profile_rows(io_rows, 1)
    normal_profile_rows, normal_mapping_rows = renumber_profile_rows(
        normal_rows,
        len(io_profile_rows) + 1,
    )
    mapping_rows = io_mapping_rows + normal_mapping_rows

    io_output = output_dir / "personas_io_drivers.csv"
    normal_output = output_dir / "personas_normal_users.csv"
    mapping_output = output_dir / "user_id_mapping.csv"
    probabilities_output = (
        args.action_probabilities_output
        or output_dir / "action_probabilities.csv"
    )

    profile_fieldnames = [*PROFILE_COLUMNS, "source_author_id"]
    write_csv(io_output, io_profile_rows, profile_fieldnames)
    write_csv(normal_output, normal_profile_rows, profile_fieldnames)
    write_csv(
        mapping_output,
        mapping_rows,
        ["simulation_user_id", "persona_user_id", "source_author_id", "username", "name"],
    )

    if args.action_probabilities_source is not None:
        remap_action_probabilities(
            args.action_probabilities_source,
            probabilities_output,
            mapping_rows,
        )
        print(f"Wrote remapped probabilities to {probabilities_output}")

    print(f"Wrote IO personas to {io_output}")
    print(f"Wrote normal personas to {normal_output}")
    print(f"Wrote user ID mapping to {mapping_output}")


if __name__ == "__main__":
    main()
