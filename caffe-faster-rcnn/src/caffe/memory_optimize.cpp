/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "caffe/memory_optimize.hpp"

#include <map>
#include <boost/make_shared.hpp>
#include <boost/algorithm/string/join.hpp>

#include "caffe/net.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
      const int64_t kNotDefined = 0;
      const int64_t kNotUsed = -1;
      const int64_t kAlwaysLive = 10000;
      const int64_t kMinimumCountForSharing = 10000;

      struct LiveRange {
        LiveRange() : defined(kNotDefined), used(kNotUsed) {}
        int64_t defined;
        int64_t used;
      };

      template <typename T>
      class Analysis : public std::map<SyncedMemory*, T> {};

      template <typename T>
      class OrderedAnalysis : public std::vector<std::pair<SyncedMemory*, T> > {};

      typedef std::pair<SyncedMemory*, LiveRange> SyncedMemoryRange;
      typedef std::vector<SyncedMemoryRange> Assignment;
      typedef std::vector<Assignment> Assignments;

      template <typename T>
      T& findOrInsert(OrderedAnalysis<T>* analysis, SyncedMemory* needle) {
        for (unsigned int i = 0; i < analysis->size(); ++i) {
          std::pair<SyncedMemory*, T>& kv = (*analysis)[i];
          if (kv.first == needle) {
            return kv.second;
          }
        }

        analysis->push_back(std::make_pair(needle, T()));
        return analysis->back().second;
      }

      template <typename Dtype>
      OrderedAnalysis<LiveRange> analyze_data(const caffe::Net<Dtype>& net) {
        // Build up the liveness analysis by walking the SyncedMemory
        // pointers attached the the blobs in the network.
        const vector<vector<Blob<Dtype>*> >& bottoms = net.bottom_vecs();
        const vector<vector<Blob<Dtype>*> >& tops = net.top_vecs();
        OrderedAnalysis<LiveRange> analysis;
        for (int64_t i = 0; i < bottoms.size(); ++i) {
          for (int64_t j = 0; j < bottoms[i].size(); ++j) {
            const Blob<Dtype>* bottom = bottoms[i][j];
            LiveRange& range = findOrInsert(&analysis, bottom->data().get());
            if (range.used == kNotUsed) {
              range.used = i;
              continue;
            }
            range.used = std::max(range.used, i);
          }
        }
        for (int64_t i = 0; i < tops.size(); ++i) {
          for (int64_t j = 0; j < tops[i].size(); ++j) {
            const Blob<Dtype>* top = tops[i][j];
            LiveRange& range = findOrInsert(&analysis, top->data().get());
            if (range.defined == kNotDefined) {
              range.defined = i;
              continue;
            }
            range.defined = std::min(range.defined, i);
          }
        }
        for (int64_t i = 0; i < net.input_blobs().size(); ++i) {
          const Blob<Dtype>* input = net.input_blobs()[i];
          findOrInsert(&analysis, input->data().get()).defined = -kAlwaysLive;
          findOrInsert(&analysis, input->data().get()).used = kAlwaysLive;
        }
        return analysis;
      }

      template <typename Dtype>
      OrderedAnalysis<LiveRange> analyze_diff(const caffe::Net<Dtype>& net) {
        // Build up the liveness analysis by walking the SyncedMemory
        // pointers attached the the blobs in the network.
        const vector<vector<Blob<Dtype>*> >& bottoms = net.bottom_vecs();
        const vector<vector<Blob<Dtype>*> >& tops = net.top_vecs();
        OrderedAnalysis<LiveRange> analysis;

        for (int64_t i = int(bottoms.size()) - 1; i >= 0; --i) {
          for (int64_t j = 0; j < bottoms[i].size(); ++j) {
            const Blob<Dtype>* bottom = bottoms[i][j];
            LiveRange& range = findOrInsert(&analysis, bottom->diff().get());
            if (range.defined == kNotDefined) {
              range.defined = -i;
              continue;
            }
            range.defined = std::min(range.defined, -i);
          }
        }
        for (int64_t i = int(tops.size()) - 1; i >= 0; --i) {
          for (int64_t j = 0; j < tops[i].size(); ++j) {
            const Blob<Dtype>* top = tops[i][j];
            LiveRange& range = findOrInsert(&analysis, top->diff().get());
            if (range.used == kNotUsed) {
              range.used = -i;
              continue;
            }
            range.used = std::max(range.used, -i);
          }
        }
        for (int64_t i = 0; i < net.output_blobs().size(); ++i) {
          const Blob<Dtype>* output = net.output_blobs()[i];
          findOrInsert(&analysis, output->diff().get()).defined = -kAlwaysLive;
          findOrInsert(&analysis, output->diff().get()).used = kAlwaysLive;
        }
        return analysis;
      }

      // Is the candidate range compatible with this assignment?
      bool isCompatible(const SyncedMemoryRange& candidate,
                        const Assignment& assignment) {
        if (candidate.second.used == kNotUsed ||
            assignment.back().second.used == kNotUsed) {
          return false;
        }
        if (candidate.first->size() <= kMinimumCountForSharing) {
          return false;
        }
        CHECK_GE(assignment.size(), 1);
        return candidate.second.defined > assignment.back().second.used;
      };

      template <typename Dtype>
      Analysis<std::vector<std::string> > blobNames(const caffe::Net<Dtype>& net,
                                                    const bool data_ref = true) {
        Analysis<std::vector<std::string> > names;
        const vector<shared_ptr<Blob<Dtype> > >& blobs = net.blobs();
        for (unsigned long i = 0; i < blobs.size(); ++i) {
          if (data_ref)
            names[blobs[i]->data().get()].push_back(net.blob_names().at(i));
          else
            names[blobs[i]->diff().get()].push_back(net.blob_names().at(i));
        }
        return names;
      }

      bool analysisComparator(const SyncedMemoryRange& a, const SyncedMemoryRange& b) {
        return a.second.used < b.second.used;
      }

// Compute an assignment of blobs to non-overlapping blobs.
      template <typename Dtype>
      Assignments assign(const Net<Dtype>& net, OrderedAnalysis<LiveRange> analysis,
                         const bool data_ref = true) {
        const Analysis<std::vector<std::string> >& names = blobNames(net, data_ref);
        std::stable_sort(analysis.begin(),
                         analysis.end(),
                         analysisComparator);
        for (unsigned int i = 0; i < analysis.size(); ++i) {
          const std::pair<SyncedMemory*, LiveRange>& kv = analysis[i];
          LOG(INFO) << boost::algorithm::join(names.at(kv.first), ", ") << ": " << kv.second.defined << "->" << kv.second.used;
        }

        Assignments assignments;
        for (unsigned int i = 0; i < analysis.size(); ++i) {
          const std::pair<SyncedMemory*, LiveRange>& candidate = analysis[i];
          bool assigned = false;
          for (unsigned int j = 0; j < assignments.size(); ++j) {
            Assignment& assignment = assignments[j];
            if (isCompatible(candidate, assignment)) {
              assignment.push_back(candidate);
              assigned = true;
              break;
            }
          }
          if (assigned) {
            continue;
          }
          Assignment tmp;
          tmp.push_back(candidate);
          assignments.push_back(tmp);
        }

        return assignments;
      }

      template <typename T>
      void logAssignmentMetrics(const OrderedAnalysis<T>& analysis,
                                const Assignments& assignments) {
        size_t beforeTotalSize = 0;
        for (unsigned int i = 0; i < analysis.size(); ++i) {
          const std::pair<SyncedMemory*, LiveRange>& kv = analysis[i];
          beforeTotalSize += kv.first->size();
        }
        size_t afterTotalSize = 0;
        for (unsigned int i = 0; i < assignments.size(); ++i) {
          const Assignment& assignment = assignments[i];
          size_t assignmentMaxSize = 0;
          for (unsigned int j = 0; j < assignment.size(); ++j) {
            const SyncedMemoryRange& kv = assignment[j];
            assignmentMaxSize = std::max(assignmentMaxSize, kv.first->size());
          }
          // LOG(INFO) << "Assignment max size: " << assignmentMaxSize;
          afterTotalSize += assignmentMaxSize;
        }

        LOG(INFO)
          << "Before: " << beforeTotalSize << ", After: " << afterTotalSize
          << ", Compression: " << 100.0 * (1.0 - afterTotalSize * 1.0 / beforeTotalSize);
      }

      template <typename Dtype>
      void applyDataAssignments(caffe::Net<Dtype>* net, const Assignments& assignments) {
        const Analysis<std::vector<std::string> >& names = blobNames(*net, true);
        Analysis<boost::shared_ptr<Blob<Dtype> > > reusedBlobs;
        for (unsigned int i = 0; i < assignments.size(); ++i) {
          const Assignment& assignment = assignments[i];
          boost::shared_ptr<Blob<Dtype> > reused = boost::make_shared<Blob<Dtype> >(1, 1, 1, 1);
          // Instantiate so blob->data() is valid.
          reused->cpu_data();
          LOG(INFO) << "Data assignment: ";
          for (unsigned int j = 0; j < assignment.size(); ++j) {
            const SyncedMemoryRange& kv = assignment[j];
            LOG(INFO) << "Blob: " << boost::algorithm::join(names.at(kv.first), ", ");
            reusedBlobs[kv.first] = reused;
          }
        }

        typedef std::vector<Blob<Dtype>*> BV;
        typedef std::vector<boost::shared_ptr<Blob<Dtype> > > SBV;

        BV& input_blobs = const_cast<BV&>(net->input_blobs());
        for (unsigned int i = 0; i < input_blobs.size(); ++i) {
          Blob<Dtype>* &blob = input_blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->data().get());
          reusedBlobs[reusedBlob->data().get()]= reusedBlob;
          blob->set_shared_data(*(reusedBlob.get()));
        }

        BV& output_blobs = const_cast<BV&>(net->output_blobs());
        for (unsigned int i = 0; i < output_blobs.size(); ++i) {
          Blob<Dtype>* &blob = output_blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->data().get());
          reusedBlobs[reusedBlob->data().get()]= reusedBlob;
          blob->set_shared_data(*(reusedBlob.get()));
        }

        for (unsigned int i = 0; i < net->top_vecs().size(); ++i) {
          BV& blobs = const_cast<BV&>(net->top_vecs()[i]);
          for (unsigned int j = 0; j < blobs.size(); ++j) {
            Blob<Dtype>* &blob = blobs[j];
            boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->data().get());
            reusedBlobs[reusedBlob->data().get()]= reusedBlob;
            blob->set_shared_data(*(reusedBlob.get()));
          }
        }

        for (unsigned int i = 0; i < net->bottom_vecs().size(); ++i) {
          BV& blobs = const_cast<BV&>(net->bottom_vecs()[i]);
          for (unsigned int j = 0; j < blobs.size(); ++j) {
            Blob<Dtype>* &blob = blobs[j];
            boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->data().get());
            reusedBlobs[reusedBlob->data().get()]= reusedBlob;
            blob->set_shared_data(*(reusedBlob.get()));
          }
        }

        SBV& blobs = const_cast<SBV&>(net->blobs());
        for (unsigned int i = 0; i < blobs.size(); ++i) {
          boost::shared_ptr<Blob<Dtype> > &blob = blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->data().get());
          blob->set_shared_data(*reusedBlob.get());
        }
      }

      template <typename Dtype>
      void applyDiffAssignments(caffe::Net<Dtype>* net, const Assignments& assignments) {
        const Analysis<std::vector<std::string> >& names = blobNames(*net, false);
        Analysis<boost::shared_ptr<Blob<Dtype> > > reusedBlobs;
        for (unsigned int i = 0; i < assignments.size(); ++i) {
          const Assignment& assignment = assignments[i];
          boost::shared_ptr<Blob<Dtype> > reused = boost::make_shared<Blob<Dtype> >(1, 1, 1, 1);
          // Instantiate so blob->data() is valid.
          reused->cpu_diff();
          LOG(INFO) << "Diff assignment: ";
          for (unsigned int j = 0; j < assignment.size(); ++j) {
            const SyncedMemoryRange& kv = assignment[j];
            LOG(INFO) << "Blob: " << boost::algorithm::join(names.at(kv.first), ", ");
            reusedBlobs[kv.first] = reused;
          }
        }

        typedef std::vector<Blob<Dtype>*> BV;
        typedef std::vector<boost::shared_ptr<Blob<Dtype> > > SBV;

        BV& input_blobs = const_cast<BV&>(net->input_blobs());
        for (unsigned int i = 0; i < input_blobs.size(); ++i) {
          Blob<Dtype>* &blob = input_blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->diff().get());
          reusedBlobs[reusedBlob->diff().get()]= reusedBlob;
          blob->set_shared_diff(*(reusedBlob.get()));
        }

        BV& output_blobs = const_cast<BV&>(net->output_blobs());
        for (unsigned int i = 0; i < output_blobs.size(); ++i) {
          Blob<Dtype>* &blob = output_blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->diff().get());
          reusedBlobs[reusedBlob->diff().get()]= reusedBlob;
          blob->set_shared_diff(*(reusedBlob.get()));
        }

        for (unsigned int i = 0; i < net->top_vecs().size(); ++i) {
          BV& blobs = const_cast<BV&>(net->top_vecs()[i]);
          for (unsigned int j = 0; j < blobs.size(); ++j) {
            Blob<Dtype>* &blob = blobs[j];
            boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->diff().get());
            reusedBlobs[reusedBlob->diff().get()]= reusedBlob;
            blob->set_shared_diff(*(reusedBlob.get()));
          }
        }

        for (unsigned int i = 0; i < net->bottom_vecs().size(); ++i) {
          BV& blobs = const_cast<BV&>(net->bottom_vecs()[i]);
          for (unsigned int j = 0; j < blobs.size(); ++j) {
            Blob<Dtype>* &blob = blobs[j];
            boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->diff().get());
            reusedBlobs[reusedBlob->diff().get()]= reusedBlob;
            blob->set_shared_diff(*(reusedBlob.get()));
          }
        }

        SBV& blobs = const_cast<SBV&>(net->blobs());
        for (unsigned int i = 0; i < blobs.size(); ++i) {
          boost::shared_ptr<Blob<Dtype> > &blob = blobs[i];
          boost::shared_ptr<Blob<Dtype> > reusedBlob = reusedBlobs.at(blob->diff().get());
          blob->set_shared_diff(*reusedBlob.get());
        }
      }

    template <typename Dtype>
    void MemoryOptimizer<Dtype>::optimizeMemory(caffe::Net<Dtype>* net) {
      net->Reshape();
      // If the net does sharing (e.g. SplitLayer), run a forward pass to
      // get the sharing setup so that it is indentified when we use the
      // SyncedMemory addresses as identifiers for def/use ranges.
      net->Forward();
      const OrderedAnalysis<LiveRange>& analysis_data = analyze_data(*net);
      const OrderedAnalysis<LiveRange>& analysis_diff = analyze_diff(*net);
      const Assignments& assignments_data = assign(*net, analysis_data, true);
      //const Assignments& assignments_diff = assign(*net, analysis_diff, false);

      applyDataAssignments(net, assignments_data);
      //applyDiffAssignments(net, assignments_diff);
      logAssignmentMetrics(analysis_data, assignments_data);
      //logAssignmentMetrics(analysis_diff, assignments_diff);
    }
  INSTANTIATE_CLASS(MemoryOptimizer);
}