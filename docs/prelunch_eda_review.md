# PathSnap pre-lunch EDA review

## Scope

Audited the supplied `Student folder.zip` without modifying its data. It contains 80 training trips and 10 student-test trips, with four expected files (`.track`, `.nodes`, `.arcs`, `.route`) for each trip. The evaluator-test partition is not included in this ZIP.

## Partition inventory

| Partition | Trips | GPS rows total | Median GPS rows/trip | Range GPS rows | Median nodes/trip | Median arcs/trip | Median route arcs/trip | Trips with any gap >5 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 1,529 | 361–19,313 | 9,608 | 19,526 | 280 | 45 |
| Student test | 10 | 19,160 | 1,420 | 262–3,958 | 6,596 | 13,145 | 250 | 2 |

Across all files audited: no non-finite GPS rows, out-of-range route arc IDs, or invalid arc endpoint node IDs were found. This is a basic structural check, not a complete geographic plausibility or graph-connectivity audit. Per-trip detail is in `trip_inventory.csv`; aggregates are in `partition_summary.csv`.

## Visual review and alignment evidence

Reviewed local overlays of the road network, GPS trace, and ordered route arcs for student-test trips `00000040` and `00000076`, plus training trip `00000004`. The GPS traces and provided route arcs follow broadly similar geographic paths in all three. The route arc sequence was directed-contiguous at 100%, 99.73%, and 100%, respectively. Sampled GPS-to-route distances had median / 90th-percentile values of 4.8 / 11.9 m, 7.1 / 15.8 m, and 8.3 / 25.9 m. These distances support broad spatial correspondence for these selected examples; they do **not** establish the correct arc label at each GPS timestamp.

The route contains far fewer arcs than the corresponding GPS track (for example, student trip `00000040`: 309 route arcs versus 2,995 GPS observations). Therefore do not align labels by row index or repeat route IDs. Resolve observation-level alignment separately using a documented monotone route-projection or sequence alignment method, handle gaps/ambiguities, and validate it across multiple trips before reporting pointwise scores.

Map images are available locally under `outputs/PathSnap_EDA/overlay_*.png`. They are review artifacts, not yet committed to the repository. Coordinates were plotted in a local trip-centred metric approximation for inspection; use geodesic or suitable per-trip projection calculations in the production pipeline.

## Immediate engineering actions

1. Finish label semantics and inspect more examples, especially long gaps and dense urban networks.
2. Implement per-trip road-segment geometry and candidate lookup; measure candidate recall/miss rate before training.
3. Build shared candidate features and use trip-grouped validation on training trips.
4. Fill `model_comparison_template.csv` only with measured results. Prioritize route continuity and plausible sequences; report classifier-only and HMM-decoded results separately once a decoder exists.

No model performance values are included here because no model training or validated point-label construction has been done yet.
