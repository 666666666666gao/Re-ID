"""Reuse the complete nine-arm CPU report without changing its scoring."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import report_native_partitioned as report
from tools import queue_native_research
from tools import analyze_correspondence_distances as diagnosis

original_compare = report.compare


def compare(matrix, dataset, control, candidate):
    result = original_compare(matrix, dataset, control, candidate)
    rows = {row['variant']: row for row in matrix['rows'] if row['dataset'] == dataset}
    scores = {}
    environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
    scorer = diagnosis.scene_scores if dataset == 'MSVR310' else diagnosis.camera_scores
    for variant in (control, candidate):
        data = report.torch.load(Path(rows[variant]['run_dir']) / 'official_distances.pt',
                                 map_location='cpu', weights_only=False)
        scores[variant] = scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                                 data['query_' + environment], data['gallery_' + environment])
    result['query_changes'] = [
        {'query_index': index, 'identity': int(identity),
         'control_ap': float(scores[control]['average_precision'][index]),
         'candidate_ap': float(scores[candidate]['average_precision'][index]),
         'control_first_rank': int(scores[control]['first_match_rank'][index]),
         'candidate_first_rank': int(scores[candidate]['first_match_rank'][index])}
        for index, identity in enumerate(data['query_ids'])]
    return result

report.panel = queue_native_research
report.compare = compare

if __name__ == '__main__':
    report.main()
