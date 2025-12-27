"""
Defines how datasets are imported into models
how raw audio files are loaded and transformed into tensors usable by the model.
"""
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.encoder import CategoricalEncoder

def prepare_dataset(csv_path):

    # Step 1 : create dataset from csv annotation
    dataset = DynamicItemDataset.from_csv(csv_path)

    # Step 2 : label encoding
    encoder = CategoricalEncoder()
    encoder.update_from_didataset(dataset, "emotion")
    dataset.add_dynamic_item(
        encoder.encode_label_torch,
        takes="emotion",
        provides="label_encoded"
    )

    # Step 3 : audio loading pipeline
    @takes("file_path")
    @provides("signal")
    def load_audio(file_path):
         """
        Reads raw audio waveform using SB helper
        """
         return read_audio(file_path)
    
    dataset.add_dynamic_item(load_audio)

    # Step 4 : output what model needs
    dataset.set_output_keys(["id","signal", "label_encoded"])

    return dataset
