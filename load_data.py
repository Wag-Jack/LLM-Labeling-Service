from datasets import load_dataset

def load_audio_datas(dataset_name="edinburghcstr/edacc", split="test"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset