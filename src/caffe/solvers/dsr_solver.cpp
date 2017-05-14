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
  Json::Value & record = this->record_;
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    line_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    abs_line_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    path_.push_back(0);
    lr_fix_.push_back(1);
    ratio_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    prev_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    diff_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    record[std::to_string(i)]["lr_fix"] = Json::Value(Json::arrayValue);
    record[std::to_string(i)]["line_norm"] = Json::Value(Json::arrayValue);
    record[std::to_string(i)]["path"] = Json::Value(Json::arrayValue);
    record[std::to_string(i)]["size"] = net_params[i]->count();
  }
  record["mean_ratio"] = Json::Value(Json::arrayValue);
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

#endif 

  SGDSolver<Dtype>::ApplyUpdate();
  
  Json::Value & record = this->record_;
  if ((this->iter_ % this->param_.dsr_sample_factor() == 0) && this->iter_ > 0)
  {
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    Dtype dsr_decay = this->param_.dsr_decay();
    Dtype sum_ratio = 0;
    int   param_count = 0;

    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        if (net_params[param_id]->count() < this->param_.dsr_min_dim())
            continue;

        Dtype line_norm = caffe_norm(net_params[param_id]->count(),line_[param_id]->cpu_data());
        sum_ratio += line_norm / path_[param_id];
        param_count += 1;
    }

    
    
    this->mean_ratio_ = sum_ratio / param_count;
    record["mean_ratio"].append(this->mean_ratio_);
  }
  
  if ((this->iter_ % 2000) == 0) {
      std::ofstream out("ratio.json");
      out << record;
      out.close();
  }
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
  
  caffe_copy(net_params[param_id]->count(),
      this->history_[param_id]->cpu_data(),
      this->update_[param_id]->mutable_cpu_data());

  // update history
  caffe_cpu_axpby(net_params[param_id]->count(), local_rate * lr_fix_[param_id],
            net_params[param_id]->cpu_diff(), momentum,
            this->history_[param_id]->mutable_cpu_data());

  // compute update: step back then over step
  caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
      this->history_[param_id]->cpu_data(), -momentum,
      this->update_[param_id]->mutable_cpu_data());

  if ((net_params[param_id]->count() < this->param_.dsr_min_dim()) || 
      (this->iter_ % this->param_.dsr_sample_factor() > 0)) {
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    return;
  }


  if (this->iter_==0) {
    caffe_copy(net_params[param_id]->count(),
        net_params[param_id]->cpu_data(),
        prev_[param_id]->mutable_cpu_data());

    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    return;
  }


  caffe_sub(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            prev_[param_id]->cpu_data(),
            diff_[param_id]->mutable_cpu_data());

  caffe_cpu_axpby(net_params[param_id]->count(), (1-dsr_decay),
            diff_[param_id]->cpu_data(), dsr_decay,
            line_[param_id]->mutable_cpu_data());

  Dtype diff_norm = caffe_norm(net_params[param_id]->count(),diff_[param_id]->cpu_data());

  path_[param_id] = path_[param_id] * dsr_decay + (1-dsr_decay) * diff_norm;

  Dtype line_norm = caffe_norm(net_params[param_id]->count(),line_[param_id]->cpu_data());

  lr_fix_[param_id] = std::pow(line_norm / (diff_norm * this->mean_ratio_) , Dtype(this->param_.dsr_power()))  ;
    
  Json::Value & record = this->record_;
  record[std::to_string(param_id)]["lr_fix"].append(lr_fix_[param_id]);
  record[std::to_string(param_id)]["line_norm"].append(line_norm);
  record[std::to_string(param_id)]["path"].append(path_[param_id]);

  caffe_copy(net_params[param_id]->count(),
      net_params[param_id]->cpu_data(),
      prev_[param_id]->mutable_cpu_data());

  caffe_copy(net_params[param_id]->count(),
      this->update_[param_id]->cpu_data(),
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
