# Reference data

`v1_reported.csv` holds the test results published by the v1.0 coursework project
(tag `v1.0-course-project`), transcribed from
[docs/legacy/v1_results/Model_Results.txt](../../docs/legacy/v1_results/Model_Results.txt).
Each row is a single unseeded training run evaluated on 100 test episodes; v1.0 reported a
standard deviation for REINFORCE only. A success is an episode that terminates with a return
of at least 200, the same definition v2 uses.

These numbers cannot be regenerated: v1.0 had no seeding and its published code differs from
the code that produced them. They are kept as data so that the replication tables and figures
are built from files rather than typed in by hand.
