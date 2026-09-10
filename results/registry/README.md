# Append-only registry of every evaluation run.
#
# One JSON object per line. Required fields:
#   run_id, timestamp, git_sha, dirty (bool), config_path, config_hash,
#   dataset, dataset_manifest_hash, method, metrics {}, notes
#
# Never edit or reorder existing lines. Corrections are new rows referencing supersedes: <run_id>.
