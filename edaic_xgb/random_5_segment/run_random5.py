"""Run the unchanged four-GPU EDAIC workflow on fixed five-segment metadata."""
import json

import runner_core as core
from prepare_random5 import digest


def main():
    root = core.ROOT
    manifest = json.loads((root/'selection_manifest.json').read_text())
    assert manifest['seed'] == 42 and manifest['eligible_patients'] == 229
    assert manifest['selected_segments'] == 1145
    assert digest(root/'selected_segments.csv') == manifest['selected_segments_sha256']
    for filename, sha in manifest['selected_metadata_sha256'].items():
        assert digest(root/'metadata'/filename) == sha, filename
    validate = core.load_and_validate

    def validate_random5():
        datasets, common = validate()
        assert len(common) == 229
        for feature, data in datasets.items():
            X, y, groups, pids, scores = data
            assert len(X) == 1145
            assert all((groups == pid).sum() == 5 for pid in pids)
        print('Random-5 validation passed: each LOPO fold has 1140 training segments and 5 test segments.')
        return datasets, common

    core.load_and_validate = validate_random5
    core.main()


if __name__ == '__main__':
    main()
