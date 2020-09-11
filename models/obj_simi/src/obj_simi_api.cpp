#include <torch/extension.h>
#include <vector>
#include <iostream>

void  obj_valid_inds(torch::Tensor mask, torch::Tensor batch_valid_inds){
    const int batch_size = mask.size(0);
    int num = batch_valid_inds.size(1);

    for(int i = 0; i < batch_size; i++){
        auto pixel_inds = mask[i].view(-1).nonzero();
        auto nonzero_num = pixel_inds.size(0);
        auto rand_inds = torch::randperm(nonzero_num);

        for (int j = 0; j < num; j++)
        {
            batch_valid_inds[i][j] = pixel_inds[j%nonzero_num].item();
        }
    }
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("obj_valid_inds", &obj_valid_inds, "obj_valid_inds");
}
