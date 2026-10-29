# Requirement checklist

Each requirement from the course's *Project Guidelines* and *Midterm Report Guidelines*, with where
it is met. "Report" means `report/report.pdf`; "NB" means `notebooks/midterm_analysis.ipynb`.

## Midterm report guidelines

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Source, rows, columns; instances x attributes >= 3M after encoding, with arithmetic; features with real-world meaning | done | Report Section 1 (91,775 x 219 = 20,098,725; raw 91,954 x 90 = 8,275,860; key-features table); NB sections 1, 5 |
| 2 | Preparation (encoding, outliers, missing values, duplicates) and why | done | Report Section 2 (12 steps, each with the alternative rejected); `outputs/tables/prep_log.csv` |
| 3 | Before-and-after observations | done | Report Section 3; `outputs/tables/before_after.csv` |
| 4 | Basic questions, novelty of the problem statements, EDA plots with one-line takeaways | done | Report Section 4: 4 basic questions with quick answers, P1/P2 boxed with P1 primary, novelty and causal-flavour paragraphs, Figures 1-4 with takeaway captions |
| 5 | 3-5 hypothesis tests, each with a) H0/H1 (H0 carries the equality), alpha, side, test, smallest difference that matters, written before testing; b) other tests considered and why not; c) statistic, d.f., p-value, decision, estimate with 95% CI, practical size | done | Report Section 5, Tests 1-4, each with parts a-c; `docs/preregistration.json` written before testing (SHA-256 printed by NB); `outputs/tables/test_results.json` |
| 6 | Link to the code and a README | done (push pending) | README.md; link github.com/Nishant7p/DataSci-Project in the report header; the group pushes the repository |
| 7 | PDF, at most 8 pages, LaTeX template, by 27 Oct 2026 11:59 pm on the classroom | done (5 pages) | `report/report.pdf` from `report/report.tex` (the course template); submission is the group's step |
| 8 | Presentation to follow | later | - |
| - | Graded on depth of digging, not on rejecting H0 | done | three designs for P1 (strata, clusters, host-level), sensitivity analyses, a null-sized P2 effect reported as such |

## Project guidelines

| Requirement | Status | Evidence |
|---|---|---|
| Raw or uncleaned data, not competition-ready | done | Inside Airbnb scrape: text prices, 15 empty columns, 179 duplicate units, HTML, invalid scores |
| Mix of nominal, discrete and continuous attributes | done | `docs/data_dictionary.md`: 40 nominal, 40 discrete, 21 continuous |
| Size floor checked before committing | done | `outputs/tables/size_check.json` |
| Cite where the data came from | done | Report Section 1; README "Data" |
| Data dictionary: type, unit, range, missingness for every attribute | done | `docs/data_dictionary.md` / `.csv` (90 raw + 11 derived) |
| Missing values, outliers, noisy records, duplicates, encoding, each with a reason | done | Report Section 2; `src/prep.py` |
| Effect in numbers, what was lost and gained | done | Report Section 3 table and paragraph |
| Split before exploring; confirmation set held out; split and seed recorded | done | Seed 558, by host, 30/70; Report Section 4; `outputs/tables/split_assignment.csv.gz` |
| EDA with takeaway captions | done | Figures 1-4 (exploration split only) |
| Obvious questions; short list of problem statements, primary marked, why not obvious | done | Report Section 4 |
| Correlation is not causation: causal flavour stated with what would justify it | done | Report Section 4 "Causal flavour"; Section 6 |
| Multiple comparisons corrected | done | Holm across T1-T4, family-wise alpha 0.05 |
| Justified / Quantified / Reproducible | done | every step has a reason; every claim a number; seeds, pinned versions, checksum, `pytest` checks that the report's numbers match the outputs |

## Only the group can supply

- The group name (red placeholder in the report header).
- Pushing the repository to github.com/Nishant7p/DataSci-Project and submitting the PDF and notebook.
