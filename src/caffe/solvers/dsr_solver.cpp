#include "caffe/dsr_solver.hpp"
#include <cmath>
#include <json/writer.h>
#include <iostream>
#include <string>

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
    path_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    ratio_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    abs_history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    abs_line_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}



template <typename Dtype>
void DSRSolver<Dtype>::ApplyUpdate() {
#if 0
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype dsr_decay = this->param_.dsr_decay();
  Json::Value & record = this->record_;
  Dtype sum_ratio = 0;
  int   param_count = 0;

  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1),
                net_params[param_id]->cpu_diff(), dsr_decay,
                line_[param_id]->mutable_cpu_data());
      
      Dtype current_diff_norm = 0; 
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), Dtype(1.0),
                net_params[param_id]->cpu_diff(), 
                net_params[param_id]->cpu_diff(), 
                Dtype(0),
                &(current_diff_norm));

      path_[param_id] = path_[param_id] * dsr_decay + std::sqrt(current_diff_norm);

      Dtype line_norm = 0;
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), Dtype(1.0),
                line_[param_id]->cpu_data(), 
                line_[param_id]->cpu_data(), 
                Dtype(0),
                &(line_norm));

      line_norm = std::sqrt(line_norm);


      ratio_[param_id] = line_norm/(path_[param_id] + this->param_.dsr_eps());
            
      if (!record.isMember(std::to_string(param_id))) {
          record[std::to_string(param_id)]["ratio"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["line"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["path"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["lr_fix"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["size"] = net_params[param_id]->count();
      }

      record[std::to_string(param_id)]["ratio"].append(ratio_[param_id]);
      record[std::to_string(param_id)]["line"].append(line_norm);
      record[std::to_string(param_id)]["path"].append(path_[param_id]);
     
      if (net_params[param_id]->count() >= this->param_.dsr_min_dim()) { 
        param_count += 1;
        sum_ratio += ratio_[param_id];
      }

  }
  
  if ((this->iter_ % 2000) == 0) {
      std::ofstream out("ratio.json");
      out << record;
      out.close();
  }

  this->mean_ratio_ = (this->mean_ratio_ * dsr_decay + sum_ratio / param_count) / (1 + dsr_decay);
#endif 
  SGDSolver<Dtype>::ApplyUpdate();
}


template <typename Dtype>
Dtype DSRSolver<Dtype>::GetParamMomentum(int param_id)
{
  return this->param_.momentum();
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void DSRSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  Dtype dsr_decay  = this->param_.dsr_decay();
  // Compute the update to history, then copy it to the parameter diff.
  //switch (Caffe::mode()) {
  //case Caffe::CPU: {
  caffe_cpu_axpby(net_params[param_id]->count(), (1-momentum),
            net_params[param_id]->cpu_diff(), momentum,
            this->history_[param_id]->mutable_cpu_data());

  caffe_abs(net_params[param_id]->count() , this->history_[param_id]->cpu_data() , 
            abs_history_[param_id]->mutable_cpu_data());
  
  caffe_cpu_axpby(net_params[param_id]->count(), (1-dsr_decay),
            this->history_[param_id]->cpu_data(), dsr_decay,
            line_[param_id]->mutable_cpu_data());
  
  caffe_cpu_axpby(net_params[param_id]->count(), (1-dsr_decay),
            abs_history_[param_id]->cpu_data(), dsr_decay,
            path_[param_id]->mutable_cpu_data());

  caffe_add_scalar(net_params[param_id]->count(), Dtype(this->param_.dsr_eps()),
                   path_[param_id]->mutable_cpu_data());
  
  caffe_abs(net_params[param_id]->count() , line_[param_id]->cpu_data() , abs_line_[param_id]->mutable_cpu_data());
  
  caffe_div(net_params[param_id]->count() , 
            abs_line_[param_id]->cpu_data() , 
            path_[param_id]->cpu_data() ,
            ratio_[param_id]->mutable_cpu_data());

  caffe_mul(net_params[param_id]->count(),
            this->history_[param_id]->cpu_data(),
            ratio_[param_id]->cpu_data(),
            update_[param_id]->mutable_cpu_data());

  caffe_cpu_axpby(net_params[param_id]->count(), local_rate / (this->param_.dsr_target_ratio() * (1 - momentum)),
            update_[param_id]->cpu_data(), Dtype(0),
            net_params[param_id]->mutable_cpu_diff());
  

    //break;
#if 0
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
#endif 
}



INSTANTIATE_CLASS(DSRSolver);
REGISTER_SOLVER_CLASS(DSR);


}
