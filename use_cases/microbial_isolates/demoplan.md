# Demo Plan

1. Initialize project: `dsagt init isolate-pipeline --agent claude`
2. Configure `dsagt_config.yaml` with API keys and embedding endpoint
3. Start session: `dsagt start isolate-pipeline`
4. Ask agent to load fastp and megahit docs into knowledge base
5. Ask agent to register fastp and megahit tools (conda bin paths)
6. Ask agent to design pipeline for preprocessing and assembling short read sequence data — give one file as example
7. After pipeline design, ask agent to run on all files in the directory
8. Ask agent to search for and use the datacard-generator skill
9. Ask agent to reconstruct the pipeline from execution records
10. Show artifacts: knowledge base collection, processed data, datacard, registered tools, pipeline script, MLflow traces
