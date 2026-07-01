from __future__ import annotations

from src.emailurgency.pipelines.dataloader import DataLoader
from src.emailurgency.pipelines.preprocess import PreProcess
from src.emailurgency.pipelines.llm_labeler import LLMLabeler

def main() -> None:
    dataloader = DataLoader()
    data = dataloader.fetch()

    preprocess = PreProcess()
    data_clean = preprocess.preprocess(data)

    llm_labeler = LLMLabeler()
    label = llm_labeler.label(data_clean)

    X_train, y_train, X_test, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    
    
    return

if __name__ == "__main__":
    main()