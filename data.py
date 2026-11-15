data_set = load_dataset("HuggingFaceFW/fineweb-edu", split= "train", streaming = True)
factual_dataset = load_dataset("wikimedia/wikipedia", "20231101.en",split = "train", streaming = True)

full_ds = interleave_datasets([data_set, factual_dataset], probabilities = [0.65, 0.35])