#include "caffe/dsr_solver.hpp"

namespace caffe {

template <typename Dtype>
void DSRSolver<Dtype>::AllocateLinePath() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  line_.clear();
  path_.clear();
  ratio_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    line_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    path_.push_back(0);
    ratio_.push_back(0);
  }
}



template <typename Dtype>
void DSRSolver<Dtype>::ApplyUpdate() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype dsr_decay = this->param_.dsr_decay();
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_cpu_axpby(net_params[param_id]->count(), 1.0,
                net_params[param_id]->cpu_diff(), dsr_decay,
                line_[param_id]->mutable_cpu_data());
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), 1.0,
                net_params[param_id]->cpu_diff(), 
                1,
                net_params[param_id]->cpu_diff(), 
                dsr_decay,
                &(path_[param_id]));

      Dtype line_norm = 0;
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), 1.0,
                line_[param_id]->cpu_diff(), 
                1,
                line_[param_id]->cpu_diff(), 
                0,
                &(line_norm));

      ratio_[param_id] = line_norm/path_[param_id];
  }

  SGDSolver<Dtype>::ApplyUpdate();
}


template <typename Dtype>
void DSRSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id] * this->param_().dsr_target_ratio() * ratio_[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              this->history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        this->history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}


}
