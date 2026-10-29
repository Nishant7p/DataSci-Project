# Do multi-listing hosts drag ratings down?
### Evidence from London Airbnb listings

**CSE558 Data Science, Monsoon 2026, IIIT-Delhi** · Midterm project: statistical inference

We ask whether Airbnb listings run by large operators (hosts with 10 or more London listings) get
different guest ratings from comparable listings run by single-listing hosts, and whether
marketing hype in a listing's description changes how *accurate* guests find it. The whole
analysis (data download, cleaning, exploration, pre-registered hypothesis tests) is one
notebook that anyone can re-run and get the same numbers.

## Team

| Name | Roll no. |
|---|---|
| Nischay Soni | 2023353 |
| Nishant Tomer | 2023355 |
| Sarthak | 2023490 |
| Syed Musa Kazim | 2023543 |
| Yash | 2023605 |

All five members contributed equally.

## Findings

**P1 (primary).** Among comparable entire homes (same borough and number of bedrooms), listings run
by large hosts rate **0.145 star lower** than listings run by single-listing hosts, and have
**3.2 times the odds** of falling below the 4.8 rating Airbnb requires for Superhost status.
Three different designs agree, and the gap is larger than the 0.10 star we set in advance as the
smallest difference that matters.

**P2.** Listings whose descriptions pile on superlatives ("perfect", "stunning", "luxury")
score **+0.008 star** higher on accuracy: statistically detectable, but far below anything a
guest could see.

| Test | Question | Method | Estimate (95% CI) | Holm p | Verdict |
|---|---|---|---|---|---|
| T1 | P1 | within-stratum gap, host-clustered SE | -0.145 star (-0.168, -0.122) | 2.3e-34 | reject H0; matters |
| T2 | P1 | Cochran-Mantel-Haenszel, rating < 4.8 | odds ratio 3.22 (2.97, 3.49) | 1.1e-183 | reject H0; matters |
| T3 | P1 | host-level Welch t-test | -0.164 star (-0.182, -0.146) | 1.0e-55 | reject H0; matters |
| T4 | P2 | stratified permutation test | +0.008 star (+0.0004, +0.016) | 0.044 | reject H0; too small to matter |

The full write-up is [`report/report.pdf`](report/report.pdf) (5 pages, course LaTeX template).

## Dataset

| | |
|---|---|
| Source | [Inside Airbnb](https://insideairbnb.com/get-the-data/), London, England, United Kingdom |
| Snapshot | compiled 17 September 2026 |
| File | [`listings.csv.gz`](https://data.insideairbnb.com/united-kingdom/england/london/2026-09-17/data/listings.csv.gz) (detailed listings, 48 MB) |
| Size | 91,954 rows x 90 columns; one row is one listing |
| SHA-256 | `8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c` |
| Licence | CC BY 4.0 |

Cite as: Inside Airbnb (2026), *London, England, United Kingdom: listings.csv.gz*, data compiled
17 September 2026, <https://insideairbnb.com/get-the-data/>.

The raw file is not committed; the notebook downloads it and checks the SHA-256 above. Inside
Airbnb keeps about 12 months of snapshots online; after that, request this snapshot from their
archive. Links to every dataset we considered are in [`docs/dataset_links.pdf`](docs/dataset_links.pdf).

## Reproduce the results

Requirements: Python 3.13 (tested with 3.13.15 on macOS), an internet connection for the 48 MB
download, about one minute of run time.

```bash
git clone https://github.com/Nishant7p/DataSci-Project.git
cd DataSci-Project
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the whole analysis from a clean kernel (writes outputs/ and docs/data_dictionary.*)
jupyter nbconvert --to notebook --execute --inplace notebooks/midterm_analysis.ipynb

# unit tests, plus checks that every key number in the report matches the outputs
python -m pytest -q
```

Every random step (the exploration/confirmation split, one-listing-per-host sampling, 10,000
permutations, 5,000 bootstrap resamples) uses seed **558**, so a re-run reproduces every table
byte for byte.

To rebuild the report PDF, run `tectonic report/report.tex`, or upload the `report/` folder to
Overleaf and compile with pdfLaTeX.

## How the analysis works

1. **Get the data.** Download the snapshot and verify its checksum.
2. **Data dictionary.** Type (nominal, discrete, continuous), unit, range and missingness of all
   90 raw and 11 derived attributes: [`docs/data_dictionary.md`](docs/data_dictionary.md).
3. **Clean.** Drop 15 empty and 7 redundant columns; parse prices (stored as text with a `$`,
   though the currency is GBP), bathrooms and amenities; flag missing values that have a
   structural cause instead of imputing them; fix impossible values; remove 179 duplicate units.
4. **Split before exploring.** Whole hosts go to one side: 30% exploration, 70% confirmation.
5. **Fence outliers and encode**, fitted on the exploration split only. The encoded table is
   91,775 x 219 = 20,098,725 cells, above the course's 3,000,000 floor.
6. **Explore** the exploration split only (Figures 1 to 4 in `outputs/figures/`).
7. **Pre-register** hypotheses, sides, tests and the smallest differences that matter in
   [`docs/preregistration.json`](docs/preregistration.json) before any test runs. The notebook
   prints the file's SHA-256 before testing.
8. **Test** on the confirmation split only. Each test is reported with the alternatives we
   considered, and a Holm correction covers all four.

## Repository structure

```
notebooks/midterm_analysis.ipynb   the full analysis, executed, with outputs
src/prep.py                        download, checksum, cleaning, split, encoding
src/stats_tests.py                 Welch, Cochran-Mantel-Haenszel, clustered fixed-effects t,
                                   stratified permutation and bootstrap, Holm
src/dictionary.py                  data dictionary specification
src/figures.py                     report figures
tests/                             unit tests and report-consistency checks (pytest)
docs/                              data dictionary, pre-registration, requirement checklist,
                                   dataset links
outputs/tables/                    every number used in the report (JSON and CSV)
outputs/figures/                   Figures 1 to 4 (PDF and PNG)
report/                            report.tex, report.pdf and its figures;
                                   template_original.tex is the untouched course template
```

## Data attribution

Data from [Inside Airbnb](https://insideairbnb.com), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
