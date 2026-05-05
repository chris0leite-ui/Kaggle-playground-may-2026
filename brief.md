# brief.md — verbatim host material

Source: kaggle CLI fallback (`kaggle competitions pages playground-series-s6e5
--content --page-name <name>`) on 2026-05-04. WebFetch on the public URL was
gated by Kaggle JS-rendering and returned only the page title.

## Description

> **Welcome to the 2026 Kaggle Playground Series!** We plan to continue in the
> spirit of previous playgrounds, providing interesting and approachable
> datasets for our community to practice their machine learning skills, and
> anticipate a competition each month.
>
> **Your Goal:** Predict whether a Formula 1 driver will pit on the next lap.

(From the "About the Tabular Playground Series" page:)

> The goal of the Tabular Playground Series is to provide the Kaggle
> community with a variety of fairly light-weight challenges that can be
> used to learn and sharpen skills in different aspects of machine learning
> and data science. The duration of each competition will generally only last
> a few weeks, and may have longer or shorter durations depending on the
> challenge. The challenges will generally use fairly light-weight datasets
> that are synthetically generated from real-world data, and will provide an
> opportunity to quickly iterate through various model and feature
> engineering ideas, create visualizations, etc.
>
> ### Synthetically-Generated Datasets
>
> Using synthetic data for Playground competitions allows us to strike a
> balance between having real-world data (with named features) and ensuring
> test labels are not publicly available. This allows us to host competitions
> with more interesting datasets than in the past. While there are still
> challenges with synthetic data generation, the state-of-the-art is much
> better now than when we started the Tabular Playground Series two years
> ago, and that goal is to produce datasets that have far fewer artifacts.
> Please feel free to give us feedback on the datasets for the different
> competitions so that we can continue to improve!

## Evaluation

> Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic)
> between the predicted probability and the observed target.
>
> ## Submission File
> For each id in the test set, you must predict a probability for the
> `PitNextLap` target. The file should contain a header and have the
> following format:
>
>     id,PitNextLap
>     439140,0.2
>     439141,0.3
>     439142,0.9
>     etc.

## Data description

> The dataset for this competition (both train and test) was inspired by
> [F1 strategy dataset](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/data).
> Feature distributions are close to, but not exactly the same, as the
> original, and we intentionally remove `Normalized_TyreLife` which makes the
> prediction trivial. Feel free to use the original dataset as part of this
> competition, both to explore differences as well as to see whether
> incorporating the original in training improves model performance.
>
> ## Files
>
> *   **train.csv** - the training set, with `PitNextLap` as target
> *   **test.csv** - the test set, used to predict the likelihood for `PitNextLap`
> *   **sample_submission.csv** - a sample submission file in the correct format

(The host page does not enumerate per-column descriptions. Files listed by
`kaggle competitions files`: `train.csv` 53.7 MB, `test.csv` 22.3 MB,
`sample_submission.csv` 1.7 MB; all dated 2026-04-23.)

## Rules

Competition-specific rules, verbatim excerpts:

> **1. COMPETITION TITLE** Playground Series - Season 6, Episode 5
> **2. COMPETITION SPONSOR** Google LLC
> **5. TOTAL PRIZES AVAILABLE** Choice of Kaggle merchandise
> **6. WINNER LICENSE TYPE** None
> **7. DATA ACCESS AND USE** Attribution 4.0 International (CC BY 4.0)

> **TEAM LIMITS**
> a. The maximum Team size is three (3).
> b. Team mergers are allowed and can be performed by the Team leader. In
> order to merge, the combined Team must have a total Submission count less
> than or equal to the maximum allowed as of the Team Merger Deadline. The
> maximum allowed is the number of Submissions per day multiplied by the
> number of days the competition has been running.

> **SUBMISSION LIMITS**
> a. You may submit a maximum of five (5) Submissions per day.
> b. You may select up to two (2) Final Submissions for judging.

> **EXTERNAL DATA AND TOOLS**
> a. You may use data other than the Competition Data ("External Data") to
> develop and test your Submissions. However, you will ensure the External
> Data is either publicly available and equally accessible to use by all
> Participants of the Competition for purposes of the competition at no cost
> to the other Participants, or satisfies the Reasonableness criteria as
> outlined in Section 2.6.b below. ...
> b. The use of external data and models is acceptable unless specifically
> prohibited by the Host. ...
> c. Automated Machine Learning Tools ("AMLT") ... Individual Participants
> and Teams may use automated machine learning tool(s) ("AMLT") (e.g.,
> Google toML, H2O Driverless AI, etc.) to create a Submission ...

Timeline (verbatim):

> * **Start Date** - May 1, 2026
> * **Entry Deadline** - Same as the Final Submission Deadline
> * **Team Merger Deadline** - Same as the Final Submission Deadline
> * **Final Submission Deadline** -  May 31, 2026
>
> All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise
> noted. The competition organizers reserve the right to update the contest
> timeline if they deem it necessary.

Prizes (verbatim):

> - 1st Place - Choice of Kaggle merchandise
> - 2nd Place - Choice of Kaggle merchandise
> - 3rd Place - Choice of Kaggle merchandise
>
> **Please note:** In order to encourage more participation from beginners,
> Kaggle merchandise will only be awarded once per person in this series. If
> a person has previously won, we'll skip to the next team.

## Host forum posts

(none on host page — Discussion tab not surfaced via the kaggle CLI; no
pinned host posts visible from the pages endpoint.)

## Notes

- Fallback used: `kaggle competitions pages playground-series-s6e5 --content
  --page-name <NAME>` for pages: `rules`, `Evaluation`, `Timeline`,
  `data-description`, `abstract`, `Prizes`, `About the Tabular Playground
  Series`. WebFetch on `kaggle.com/competitions/playground-series-s6e5/...`
  returned only `<title>` because the body is JS-rendered.
- File listing from `kaggle competitions files playground-series-s6e5`.
- Host competition admin metadata (cited by the mpwolke notebook):
  authors = Yao Yan, Walter Reade, Elizabeth Park.
- Original dataset reference is allowed: `aadigupta1601/f1-strategy-dataset-pit-stop-prediction`.
  Host explicitly removed `Normalized_TyreLife` because it makes prediction
  trivial — DO NOT reintroduce that column from the original.
