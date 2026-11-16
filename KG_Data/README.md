# Institution bootstrap data

`Universities.csv` is the original 1,303-row UGC file supplied with this workspace;
it has not been edited. The importer uses its real column names, records the file
SHA-256 and row number, and labels its provenance `UGC_USER_SUPPLIED`. Its download
date and exact source URL were not supplied and are not invented.

`supplementary_institutions.csv` contains **32 manually normalized records**:

- All 23 IITs from the [government ISTI directory](https://www.indiascienceandtechnology.gov.in/organisations/centres-of-higher-learning/iits), including its second and third pages.
- [IISc](https://www.iisc.ac.in/about/general-information/contact/) and [ISI](https://asd.isical.ac.in/static/contact).
- Seven additional universities, completing coverage of the 12 notified institutions in the [UGC IoE list](https://ioe.ugc.ac.in/Home/ListofIOE) accessed on 2026-10-24. Four IITs and IISc already occur in the first two groups.

The CSV is a **review draft**, not an assertion that it has been human verified.
Every new record has `verification_status=NEEDS_REVIEW`. `iit_source_records.json`
preserves the names, state fields and website links extracted from the government
directory. Some of those website links are legacy URLs; they are preserved as
published. A blank address means the extractor did not capture one.

`ioe_status=NOTIFIED` means included in the referenced UGC list. A recommendation
or letter of intent is not treated as notification. `NOT_LISTED_IN_REFERENCED_IOE_LIST`
does not assert that an institute lacks other recognition. A blank notification
date means not transcribed. The 12 IoE records are IISc, IIT Bombay, IIT Delhi,
IIT Madras, IIT Kharagpur, Delhi University, University of Hyderabad, BHU, BITS
Pilani, MAHE, O P Jindal Global University and Shiv Nadar University.

Names and pipe-separated aliases were normalized editorially. City fields refer
to the primary institution/campus locality, not verified street addresses. Review
these fields against the per-row sources. ISI branches, overseas campuses, all
NITs, all IIITs and corporate labs are not an exhaustive part of this supplement.
In particular, an Indian parent organization does not make an overseas campus an
Indian affiliation. Bare IIT/NIT/ISI/VIT aliases are intentionally absent.

To use the draft without approving it:

```bash
.venv/bin/python -m rag.cli bootstrap KG_Data/Universities.csv KG_Data/supplementary_institutions.csv
```

Pending records are retained for inspection but excluded from trusted entity
resolution, organization vector search and geographic filters. If a pending
supplement duplicates a trusted UGC canonical name, its proposed aliases and
location are stored as a `REVIEW_RECORD`; they cannot alter trusted evidence.

After your review, copy the supplement to a reviewed CSV, correct any facts, and
set `verification_status=TRUSTED` only for approved rows. Import that file with the
same command and run `rag.cli index`. Keep `entity_id` unchanged across renames.
Reingest affected paper JSON to resolve previously unresolved affiliations.
Full candidate approval/merge tooling and automatic rewiring are deferred.
