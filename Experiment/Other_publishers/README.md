# Publisher-Agnostic Design and Experimental Validation

## run_2026-01-10.session

For compliance with copyright and access regulations, the crawler component of MOFh6 was initially evaluated using literature from the mainstream academic publishers that provide broad and well-documented coverage of MOF structures. These documents were used solely to validate the stability and correctness of the document acquisition stage.

Importantly, the core functionality of MOFh6—Task I: synthesis process analysis—is inherently publisher-agnostic. This task does not rely on publisher-specific metadata, formatting rules, or proprietary interfaces. Instead, it operates directly on user-provided literature files, treating each document as a standalone input for synthesis information extraction.

To verify this design principle, additional experiments were conducted using literature obtained through legitimate access from multiple publishers beyond the initial evaluation scope (Table 1). All documents were processed using the same end-to-end workflow, without any architectural modification, rule adjustment, or prompt redesign.

### Table 1 **CCDC codes and DOI of MOFs reported by four publishers, independent of the MOFh6 TDM service**

| CCDC code of MOFs | Publisher                                                    | DOI                                                          |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| AGAWUA            | Proceedings of the National Academy of  Sciences of the United States of America (PNAS) | 10.1073/pnas.082051899                                       |
| EFARAE            | International  Union of Crystallography (IUCr)               | 10.1107/S0108270101021515                                    |
| EGEXOD01          | Trans  Tech Publications (Scientific.Net)                    | 10.4028/[www.scientific.net/AMR.282-283.96](file:///Users/linzuhong/Desktop/www.scientific.net/AMR.282-283.96) |
| FIRTEH            | American  Association for the Advancement of Science (AAAS)  | 10.1126/science.1246423                                      |
| WALCIX            | American  Association for the Advancement of Science (AAAS)  | 10.1126/science.1194237                                      |