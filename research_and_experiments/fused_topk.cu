#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>

__device__ void warp_maxer(T& val,int & idx){
    for(int i=16;i>0;i/=2){
        T next_thread_val= __shfl_down_sync(0xFFFFFFFF, val, i);
        int next_idx= __shfl_down_sync(0xFFFFFFFF, idx, i);

        if(next_thread_val>val){
            val=next_thread_val;
            idx=next_idx;
        }
    }
}

__global__ void fused_topk(const float* __restrict__ logits, int* __restrict__ output, int vocab_size){
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    const float* my_logits = logits+(blockID*vocab_size);
    float max_val = -1e20f;
    int max_idx=-1;

    for(int i=threadID;i<vocab_size;i+=blockDim.x){
        float val = my_logits[i];
        if(val>max_val){
            max_val=val;
            max_idx=i;
        }
    }

    warp_maxer(max_val, max_idx);

    
    static __shared__ float shared_vals[32]; 
    static __shared__ int shared_idxs[32];
    
    int lane = threadID % 32;    
    int warp_id = threadID / 32;

    if (lane == 0) {
        shared_vals[warp_id] = max_val;
        shared_idxs[warp_id] = max_idx;
    }
    
    __syncthreads();

    if (threadID < 32) {
        float val = (threadID < (blockDim.x / 32)) ? shared_vals[threadID] : -1e20f;
        int idx = (threadID < (blockDim.x / 32)) ? shared_idxs[threadID] : -1;

        warp_maxer(val, idx);

        if (threadID == 0) {
            output[blockID] = idx;
        }
    }
}

torch::Tensor run_fused_top1(torch::Tensor logits) {
    auto output = torch::empty({logits.size(0)}, torch::kInt32).to(logits.device());
    int vocab_size = logits.size(1);
    int batch_size = logits.size(0);
    
    fused_topk<<<batch_size, 1024>>>(
        logits.data_ptr<float>(),
        output.data_ptr<int>(),
        vocab_size
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_top1", &run_fused_top1, "Custom Whisper Top-1 Kernel");
}