import tensorflow as tf
from cgcnn.data_tf import CIFData, collate_pool, Dataloader
from cgcnn.model_tf import CrystalGraphConvNet


if __name__=="__main__":
    dataset = CIFData("data/sample-regression/")

    batched = Dataloader(dataset, batch_size=1)

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    cgcnn_m = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                            atom_fea_len=64,
                            n_conv=3,
                            h_fea_len=128,
                            n_h=1)

    cgcnn_m.compile(optimizer='adam', loss='mean_squared_error',metrics=['mse', 'mae'], run_eagerly=True)



    history = cgcnn_m.fit(batched, epochs=3, use_multiprocessing=False)