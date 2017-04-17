// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include <cstdio>
#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const Dtype box_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];


    int box_width = bottom_rois[3] - bottom_rois[1];
    int box_height = bottom_rois[4] - bottom_rois[2];
    int box_cntr_x = bottom_rois[1] + box_width / 2;
    int box_cntr_y = bottom_rois[2] + box_height / 2;

    int roi_start_w = round((box_cntr_x - box_width * box_scale) * spatial_scale);
    int roi_start_h = round((box_cntr_y - box_height * box_scale) * spatial_scale);
    int roi_end_w = round((box_cntr_x + box_width * box_scale) * spatial_scale);
    int roi_end_h = round((box_cntr_y + box_height * box_scale) * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
__device__ Dtype get_overlap(const Dtype a1, const Dtype b1,
                        const Dtype a2, const Dtype b2) {
    return max(min(b1, b2) - max(a1, a2), Dtype(0));
}

template <typename Dtype>
__global__ void ROIPoolForwardBilinear(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const Dtype box_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    Dtype box_width = bottom_rois[3] - bottom_rois[1] + 1;
    Dtype box_height = bottom_rois[4] - bottom_rois[2] + 1;
    Dtype box_cntr_x = bottom_rois[1] + box_width * static_cast<Dtype>(0.5);
    Dtype box_cntr_y = bottom_rois[2] + box_height * static_cast<Dtype>(0.5);

    Dtype roi_start_w = (box_cntr_x - box_width * box_scale) * spatial_scale;
    Dtype roi_start_h = (box_cntr_y - box_height * box_scale) * spatial_scale;
    Dtype roi_end_w = (box_cntr_x + box_width * box_scale) * spatial_scale;
    Dtype roi_end_h = (box_cntr_y + box_height * box_scale) * spatial_scale;

    // Clip roi to bottom size
    roi_start_w = min(max(roi_start_w, Dtype(0)), static_cast<Dtype>(width - 1));
    roi_end_w = min(max(roi_end_w, Dtype(0)), static_cast<Dtype>(width));
    roi_start_h = min(max(roi_start_h, Dtype(0)), static_cast<Dtype>(height - 1));
    roi_end_h = min(max(roi_end_h, Dtype(0)), static_cast<Dtype>(height));

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1));
    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1));
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
    const int sub_bin_num_h = ceil(bin_size_h);
    const int sub_bin_num_w = ceil(bin_size_w);
    const Dtype sub_bin_size_h = bin_size_h / sub_bin_num_h;
    const Dtype sub_bin_size_w = bin_size_w / sub_bin_num_w;

    Dtype hstart = roi_start_h + static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = roi_start_w + static_cast<Dtype>(pw) * bin_size_w;

    bottom_data += (roi_batch_ind * channels + c) * height * width;

    int maxidx = -1;
    Dtype maxval = -FLT_MAX;
    for (int h_indx = 0; h_indx < sub_bin_num_h; ++h_indx) {
      for (int w_indx = 0; w_indx < sub_bin_num_w; ++w_indx) {
        //int bottom_index = h * width + w;
        const Dtype sub_cx = wstart + (w_indx + Dtype(0.5)) * sub_bin_size_w;
        const Dtype sub_cy = hstart + (h_indx + Dtype(0.5)) * sub_bin_size_h;

        int x, y;
        Dtype w = 0, value = 0;

        x = floor(sub_cx), y = floor(sub_cy);
        if (x >= 0 && y >= 0 && x < width && y < height) {
            w = (1 - (sub_cx - x)) * (1 - (sub_cy - y));
            value += w * bottom_data[y * width + x];
        }

        x = floor(sub_cx) + 1, y = floor(sub_cy);
        if (x >= 0 && y >= 0 && x < width && y < height) {
            w = (1 - (x - sub_cx)) * (1 - (sub_cy - y));
            value += w * bottom_data[y * width + x];
        }

        x = floor(sub_cx), y = floor(sub_cy) + 1;
        if (x >= 0 && y >= 0 && x < width && y < height) {
            w = (1 - (sub_cx - x)) * (1 - (y - sub_cy));
            value += w * bottom_data[y * width + x];
        }

        x = floor(sub_cx) + 1, y = floor(sub_cy) + 1;
        if (x >= 0 && y >= 0 && x < width && y < height) {
            w = (1 - (x - sub_cx)) * (1 - (y - sub_cy));
            value += w * bottom_data[y * width + x];
        }

        if (value > maxval) {
            maxval = value;
            maxidx = h_indx * sub_bin_num_w + w_indx;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}


template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();
  int* argmax_data = max_idx_.mutable_gpu_data();

  if (use_bilinear_interpolation_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForwardBilinear<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, spatial_scale_, 0.5 * box_scale_, channels_, height_, width_,
          pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  } else {

      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, spatial_scale_, 0.5 * box_scale_, channels_, height_, width_,
          pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  }

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const Dtype box_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int box_width = offset_bottom_rois[3] - offset_bottom_rois[1] + 1;
      int box_height = offset_bottom_rois[4] - offset_bottom_rois[2] + 1;
      int box_cntr_x = offset_bottom_rois[1] + box_width / 2;
      int box_cntr_y = offset_bottom_rois[2] + box_height / 2;
      int roi_start_w = round((box_cntr_x - box_width * box_scale) * spatial_scale);
      int roi_start_h = round((box_cntr_y - box_height * box_scale) * spatial_scale);
      int roi_end_w = round((box_cntr_x + box_width * box_scale) * spatial_scale);
      int roi_end_h = round((box_cntr_y + box_height * box_scale) * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }

    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void ROIPoolBackwardBilinear(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const Dtype box_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    const Dtype wf = w;
    const Dtype hf = h;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype box_width = offset_bottom_rois[3] - offset_bottom_rois[1] + 1;
      Dtype box_height = offset_bottom_rois[4] - offset_bottom_rois[2] + 1;
      Dtype box_cntr_x = offset_bottom_rois[1] + box_width * static_cast<Dtype>(0.5);
      Dtype box_cntr_y = offset_bottom_rois[2] + box_height * static_cast<Dtype>(0.5);
      Dtype roi_start_w = (box_cntr_x - box_width * box_scale) * spatial_scale;
      Dtype roi_start_h = (box_cntr_y - box_height * box_scale) * spatial_scale;
      Dtype roi_end_w = (box_cntr_x + box_width * box_scale) * spatial_scale;
      Dtype roi_end_h = (box_cntr_y + box_height * box_scale) * spatial_scale;

      // Clip roi to bottom size
      roi_start_w = min(max(roi_start_w, Dtype(0)), static_cast<Dtype>(width - 1));
      roi_end_w = min(max(roi_end_w, Dtype(0)), static_cast<Dtype>(width));
      roi_start_h = min(max(roi_start_h, Dtype(0)), static_cast<Dtype>(height - 1));
      roi_end_h = min(max(roi_end_h, Dtype(0)), static_cast<Dtype>(height));

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (wf >= (floor(roi_start_w)) &&
                           wf <= (ceil(roi_end_w)) &&
                           hf >= (floor(roi_start_h)) &&
                           hf <= (ceil(roi_end_h)));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(1));
      Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(1));
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
      const int sub_bin_num_h = ceil(bin_size_h);
      const int sub_bin_num_w = ceil(bin_size_w);
      const Dtype sub_bin_size_h = bin_size_h / sub_bin_num_h;
      const Dtype sub_bin_size_w = bin_size_w / sub_bin_num_w;

      int phstart = floor((hf - roi_start_h) / bin_size_h);
      int phend = ceil((hf - roi_start_h + 1) / bin_size_h);
      int pwstart = floor((wf - roi_start_w) / bin_size_w);
      int pwend = ceil((wf - roi_start_w + 1) / bin_size_w);


      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          const int indx = offset_argmax_data[ph * pooled_width + pw];
          const int w_indx = indx % sub_bin_num_w;
          const int h_indx = indx / sub_bin_num_w;

          const Dtype hstart = roi_start_h + static_cast<Dtype>(ph) * bin_size_h;
          const Dtype wstart = roi_start_w + static_cast<Dtype>(pw) * bin_size_w;
          const Dtype sub_cx = wstart + (w_indx + Dtype(0.5)) * sub_bin_size_w;
          const Dtype sub_cy = hstart + (h_indx + Dtype(0.5)) * sub_bin_size_h;

          int x, y;
          Dtype wd = 0;

          x = floor(sub_cx), y = floor(sub_cy);
          if (x == w && y == h) {
              wd = (1 - (sub_cx - x)) * (1 - (sub_cy - y));
              gradient += wd * offset_top_diff[ph * pooled_width + pw];
          }

//          if (c == 113 && roi_n % 17 == 3) {
//            printf("Roi: %f %f %f %f\n%f %f %d %d %f %f\n%d %d %d %d\n%d %d %d %d\n", roi_start_w, roi_start_h, roi_end_w, roi_end_h,
//            bin_size_w, bin_size_h, sub_bin_num_w, sub_bin_num_h, sub_bin_size_w, sub_bin_size_h,
//            w, h, x, y,
//            pwstart, phstart, pwend, phend);
//          }

          x = floor(sub_cx) + 1, y = floor(sub_cy);
          if (x == w && y == h) {
              wd = (1 - (x - sub_cx)) * (1 - (sub_cy - y));
              gradient += wd * offset_top_diff[ph * pooled_width + pw];
          }

          x = floor(sub_cx), y = floor(sub_cy) + 1;
          if (x == w && y == h) {
              wd = (1 - (sub_cx - x)) * (1 - (y - sub_cy));
              gradient += wd * offset_top_diff[ph * pooled_width + pw];
          }

          x = floor(sub_cx) + 1, y = floor(sub_cy) + 1;
          if (x == w && y == h) {
              wd = (1 - (x - sub_cx)) * (1 - (y - sub_cy));
              gradient += wd * offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
//    if (c % 7 == 3)
//        printf("%d %d %d | %f\n", w, h, c, gradient);
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  const int* argmax_data = max_idx_.gpu_data();

  if (use_bilinear_interpolation_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackwardBilinear<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, argmax_data, top[0]->num(), spatial_scale_, 0.5 * box_scale_, channels_,
          height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  } else {

      // NOLINT_NEXT_LINE(whitespace/operators)
      ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, argmax_data, top[0]->num(), spatial_scale_, 0.5 * box_scale_, channels_,
          height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  }

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);

}  // namespace caffe
