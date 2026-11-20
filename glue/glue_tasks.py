"""Pinned GLUE task contracts. Importing the catalog requires only the standard library."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    instruction: str
    fields: tuple
    labels: tuple
    tokens: tuple
    train: int
    validation: dict

    def prompt(self, row):
        return self.instruction + ": " + "\n".join(row[k] for k in self.fields) + "\nresult: "


SEMANTIC = "classify the semantic similarity of the text"
TASKS = {
    "mnli": Task(SEMANTIC, ("premise","hypothesis"), ("entailment","neutral","contradiction"),
                 ((3,35,5756,297,1),(7163,1),(27252,1)), 392702,
                 {"validation_matched":9815,"validation_mismatched":9832}),
    "sst2": Task("classify the sentiment of the text", ("sentence",), ("negative","positive"),
                 ((2841,1),(1465,1)), 67349, {"validation":872}),
    "cola": Task("classify the grammaticality of the text", ("sentence",), ("unacceptable","acceptable"),
                 ((29452,1),(9961,1)), 8551, {"validation":1043}),
    "qnli": Task("classify the semantic similarity of the question and the sentence", ("question","sentence"),
                 ("entailment","not_entailment"), ((3,35,5756,297,1),(59,834,35,5756,297,1)),
                 104743, {"validation":5463}),
    "mrpc": Task(SEMANTIC, ("sentence1","sentence2"), ("different","equivalent"), ((315,1),(7072,1)),
                 3668, {"validation":408}),
}
