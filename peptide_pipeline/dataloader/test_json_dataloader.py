from dataloader_json import DataLoader as JSONDataLoader

def test():
    loader = JSONDataLoader()
    loader.load_data("ai_training_peptides.json", columns=["sequence", "cathionicity"])
    data = loader.get_data()
    print(data.columns)
    print(data.head())
    assert not data.empty, "Data should not be empty"
    assert "sequence" in data.columns, "sequence column should be present"
    assert "cathionicity" in data.columns, "cathionicity column should be present"
    print("All tests passed!")

test()