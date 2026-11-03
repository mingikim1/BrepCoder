"""Constants shared by the BrepCoder Phase 2 modules."""

# Label value ignored by the cross-entropy loss.
IGNORE_INDEX = -100

# Placeholder id written into `input_ids` where the B-rep node embeddings are inserted.
BREP_TOKEN_INDEX = -300

# Special tokens added to the LLM tokenizer.
DEFAULT_BREP_TOKEN = "<brep>"
DEFAULT_BREP_START_TOKEN = "<brep_start>"
DEFAULT_BREP_END_TOKEN = "<brep_end>"

# Prompt fed to the LLM; <brep> is replaced by the projected B-rep node embeddings.
BREP_PROMPT = f"{DEFAULT_BREP_START_TOKEN}{DEFAULT_BREP_TOKEN}{DEFAULT_BREP_END_TOKEN}"

DEFAULT_MAX_LENGTH = 2048
