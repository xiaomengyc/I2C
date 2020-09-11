int sp_segment_func(const THCudaTensor *heatmap, const THCudaLongTensor *superpixel,const THCudaLongTensor *segments, int biggest_seg, float th_high, float th_low);
int sp_atten_func(const THCudaTensor *heatmap, const THCudaLongTensor *superpixel, const THCudaLongTensor *segments, int biggest_seg);
