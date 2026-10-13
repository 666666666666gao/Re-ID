"""A standard-library feature-space witness, not a model or ReID experiment."""
import argparse
import json
import math
from pathlib import Path


def statistics(features):
    means = [math.fsum(column) / len(features) for column in zip(*features)]
    variances = [
        math.fsum((row[j] - mean) ** 2 for row in features) / len(features)
        for j, mean in enumerate(means)
    ]
    return means, variances


def batch_norm(features, means, variances, gamma, epsilon):
    return [
        [(value - means[j]) * gamma[j] / math.sqrt(variances[j] + epsilon)
         for j, value in enumerate(row)]
        for row in features
    ]


def logits(features, weights):
    return [[math.fsum(x * w for x, w in zip(row, weight))
             for weight in weights] for row in features]


def smoothed_ce(scores, labels, smoothing):
    losses = []
    for row, label in zip(scores, labels):
        maximum = max(row)
        log_sum = maximum + math.log(math.fsum(math.exp(x - maximum) for x in row))
        losses.append(math.fsum(
            ((1.0 - smoothing) * (j == label) + smoothing / len(row)) * (log_sum - x)
            for j, x in enumerate(row)
        ))
    return math.fsum(losses) / len(losses)


def raw_distances(features):
    # The 1e-12 squared-distance clamp mirrors the inspected loss's self-distance floor.
    # Difference-based binary64 arithmetic is not a replay of its PyTorch/CUDA kernels.
    return [[math.sqrt(max(math.fsum((x - y) ** 2 for x, y in zip(a, b)), 1e-12))
             for b in features] for a in features]


def hard_triplet(distances, labels):
    pairs = []
    for i, row in enumerate(distances):
        positive = max((j for j in range(len(labels)) if labels[j] == labels[i]),
                       key=lambda j: row[j])
        negative = min((j for j in range(len(labels)) if labels[j] != labels[i]),
                       key=lambda j: row[j])
        pairs.append({"anchor": i, "positive": positive, "negative": negative,
                      "soft_margin": math.log1p(math.exp(row[positive] - row[negative]))})
    return math.fsum(pair["soft_margin"] for pair in pairs) / len(pairs), pairs


def max_difference(left, right):
    return max(abs(x - y) for a, b in zip(left, right) for x, y in zip(a, b))


def cosine_query(features):
    unit = [[x / math.sqrt(math.fsum(v * v for v in row)) for x in row]
            for row in features]
    scores = {j: math.fsum(x * y for x, y in zip(unit[0], unit[j]))
              for j in range(1, len(features))}
    ranking = sorted(scores, key=scores.get, reverse=True)
    return {"query": 0, "positive": 1, "cosine_scores": scores,
            "ranking": ranking, "positive_rank": ranking.index(1) + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    features = [[1.0, 0.0], [0.8, 0.6], [1.5, -0.1], [1.6, -0.2]]
    shift = [0.0, 2.0]
    translated = [[x + b for x, b in zip(row, shift)] for row in features]
    labels = [0, 0, 1, 1]
    gamma = [1.2, 0.8]
    weights = [[0.7, -0.2], [-0.4, 0.9]]
    epsilon, smoothing = 1e-5, 0.1
    original_stats, translated_stats = statistics(features), statistics(translated)
    original_bn = batch_norm(features, *original_stats, gamma, epsilon)
    translated_bn = batch_norm(translated, *translated_stats, gamma, epsilon)
    original_logits, translated_logits = logits(original_bn, weights), logits(translated_bn, weights)
    original_ce = smoothed_ce(original_logits, labels, smoothing)
    translated_ce = smoothed_ce(translated_logits, labels, smoothing)
    original_distances, translated_distances = raw_distances(features), raw_distances(translated)
    original_triplet, original_pairs = hard_triplet(original_distances, labels)
    translated_triplet, translated_pairs = hard_triplet(translated_distances, labels)
    # A fixed-statistics evaluation contrast tests the train-mode assumption.
    fixed_bn = batch_norm(translated, *original_stats, gamma, epsilon)
    fixed_ce = smoothed_ce(logits(fixed_bn, weights), labels, smoothing)
    before, after = cosine_query(features), cosine_query(translated)
    errors = {
        "train_bn_max_abs": max_difference(original_bn, translated_bn),
        "train_logits_max_abs": max_difference(original_logits, translated_logits),
        "train_ce_abs": abs(original_ce - translated_ce),
        "raw_distances_max_abs": max_difference(original_distances, translated_distances),
        "raw_triplet_abs": abs(original_triplet - translated_triplet),
    }
    assert all(value <= 1e-12 for value in errors.values())
    assert [(p["positive"], p["negative"]) for p in original_pairs] == [
        (p["positive"], p["negative"]) for p in translated_pairs]
    assert before["positive_rank"] == 3 and after["positive_rank"] == 1
    assert abs(original_ce - fixed_ce) > 0.1
    result = {
        "status": "FEATURE_SPACE_MATHEMATICAL_WITNESS_COMPLETE",
        "scope": "Four synthetic vectors, binary64 standard-library arithmetic; not a trained model or ReID result",
        "inputs": {"features": features, "translation": shift, "labels": labels,
                   "bn_gamma": gamma, "bn_beta": [0.0, 0.0], "epsilon": epsilon,
                   "classifier_weights": weights, "classifier_bias": None, "label_smoothing": smoothing},
        "training_geometry": {
            "means_before": original_stats[0], "means_after": translated_stats[0],
            "variances_before": original_stats[1], "variances_after": translated_stats[1],
            "ce_before": original_ce, "ce_after": translated_ce,
            "triplet_before": original_triplet, "triplet_after": translated_triplet,
            "mined_pairs_before": original_pairs, "mined_pairs_after": translated_pairs,
            "invariance_errors": errors, "witness_tolerance": 1e-12},
        "fixed_eval_statistics_contrast": {
            "ce_before": original_ce, "ce_after": fixed_ce,
            "note": "Same frozen statistics; arbitrary translation no longer cancels"},
        "cosine_comparison": {"before": before, "after": after,
            "reverse_translation_note": "Reversing this translation changes the positive rank from 1 to 3; neither direction is an automatic fix"},
        "limitations": [
            "An output-level symmetry does not prove a reachable independent parameter-flat direction",
            "No actual PyTorch forward, autograd, neural model, dataset, checkpoint or optimizer was used",
            "No real common component was classified as noise, identity information, or a cause of errors",
            "No test-mean centering, new loss/head, method deployment or training is justified by this witness alone"],
        "neural_forward_calls": 0, "optimizer_updates": 0, "new_training_started": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": errors,
                      "positive_rank_before": before["positive_rank"],
                      "positive_rank_after": after["positive_rank"]}))


if __name__ == "__main__":
    main()
