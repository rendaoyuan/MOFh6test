# **Multiple synthesis routes**

## Same MOF across sources: run_2026-01-11.session

Taking **MIL-53** as an example, MOFh6 can first be queried to identify all MOFs that share the same **synonyms** (e.g., MIL-53 family variants).

Based on the returned results, MOFh6 collects the corresponding **CCDC codes** associated with these synonym-matched MOFs.

## Several MOFs within one paper: run_2026-01-12.session

Once the relevant CCDC codes are identified, MOFh6 proceeds to **precisely retrieve the synthesis methods** of the target MOFs by tracing each CCDC code back to its original literature sources.


- This two-step strategy, **synonym-based CCDC discovery followed by CCDC-anchored synthesis extraction**, ensures both comprehensive coverage of structurally related MOFs and accurate acquisition of synthesis information.
