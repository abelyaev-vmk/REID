/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef CAFFE_MEMORY_OPTIMIZE_H_
#define CAFFE_MEMORY_OPTIMIZE_H_

namespace caffe {
  template <typename Dtype>
  class Net;
}

namespace caffe {

  template<typename Dtype>
  class MemoryOptimizer {
    public:

    static void optimizeMemory(caffe::Net<Dtype> *net);
  };
}

#endif  // CAFFE_MEMORY_OPTIMIZE_H_