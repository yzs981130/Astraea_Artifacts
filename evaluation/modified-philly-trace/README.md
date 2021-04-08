# Modification for Philly trace

The original Philly trace is located at https://github.com/msr-fiddle/philly-traces

- [philly_two_weeks.csv](./philly_two_weeks.csv): The two-week period with high utilization for the evaluation.
- [name-production.lst](./name-production.lst) shows the tenants list in the trace.
- [share-production.csv](./share-production.csv) shows the weight of each tenant, inferred from [cluster_job_log](https://github.com/msr-fiddle/philly-traces/blob/master/README.md#cluster_job_log) and [cluster_machine_list](https://github.com/msr-fiddle/philly-traces/blob/master/README.md#cluster_machine_list).
