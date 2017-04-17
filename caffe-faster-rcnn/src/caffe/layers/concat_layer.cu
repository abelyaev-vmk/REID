#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Dtype>
__global__ void ConcatAutoAlignment(const int nthreads, const bool forward, 
                                    const Dtype* in_data, Dtype* out_data,
                                    const int num_concats,
                                    const int bottom_height, const int bottom_width,
                                    const int bottom_concat_axis,
                                    const int top_height, const int top_width,
                                    const int top_concat_axis,
                                    const int top_offset_concat_axis) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = bottom_height * bottom_width * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index1 = index % total_concat_size;
    const int concat_channel = concat_index1 / (bottom_width * bottom_height);
    const int concat_index2 = concat_index1 % (bottom_width * bottom_height);
    const int concat_row = concat_index2 / bottom_width;
    const int concat_col = concat_index2 % bottom_width;

    const int top_index = concat_col + concat_row * top_width +
                          (concat_num * top_concat_axis + top_offset_concat_axis + concat_channel) * 
                          (top_height * top_width);
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_input_size = bottom[i]->count(concat_axis_ + 1);
    const int bottom_concat_size = bottom_concat_axis * bottom_input_size;
    const int nthreads = bottom_concat_size * num_concats_;
    if (allow_pad_ == 0) {
      Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, bottom_data, kForward, num_concats_, concat_input_size_,
          top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    } else {
      const int bottom_height = bottom[i]->shape(2);
      const int bottom_width = bottom[i]->shape(3);
      const int top_height = top[0]->shape(2);
      const int top_width = top[0]->shape(3);

      ConcatAutoAlignment<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, kForward, bottom_data, top_data, num_concats_, 
          bottom_height, bottom_width, bottom_concat_axis,
          top_height, top_width, top_concat_axis, offset_concat_axis);
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

template <typename Dtype>
void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = false;
  for (int i = 0; i < bottom.size(); ++i) {
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_input_size = bottom[i]->count(concat_axis_ + 1);
      const int bottom_concat_size = bottom_concat_axis * bottom_input_size;
      const int nthreads = bottom_concat_size * num_concats_;
      if (allow_pad_ == 0) {
        Concat<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, top_diff, kForward, num_concats_, concat_input_size_,
            top_concat_axis, bottom_concat_axis, offset_concat_axis, bottom_diff);
      } else {
        const int bottom_height = bottom[i]->shape(2);
        const int bottom_width = bottom[i]->shape(3);
        const int top_height = top[0]->shape(2);
        const int top_width = top[0]->shape(3);

        ConcatAutoAlignment<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, kForward, top_diff, bottom_diff, num_concats_, 
            bottom_height, bottom_width, bottom_concat_axis,
            top_height, top_width, top_concat_axis, offset_concat_axis);
      }
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);

}  // namespace caffe
